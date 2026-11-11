#!/usr/bin/env python3
"""Detached CPU-only deployment of the frozen Mask R-CNN replication analysis."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parent.parent
ENV=ROOT/'.runtime_gpu2_native_v2/conda'
NATIVE_SHA='bac7367518910174451e16ed6eb8d1e8b8214c0f8b2fd81da3100305559a257a'
INPUTS={
 'CarDD_COCO/annotations/instances_val2017.json':'25d445e2da4d86a14390d419d8147a1ace97f8f71f2e37cd2e7c6be79398a8d2',
 'output_offline/R01_MVA_MaskRCNN_R50_20261017_gpu1_v1/seed42/inference/coco_instances_results.json':'645edf384c65b30871a8bf1eeba4959aa7e4fe8ab7d3cf6328e83621fc4ec977',
 'output_offline/R01_MVA_MaskRCNN_seed3407_20261026_gpu0_resume_v3/stage/inference/coco_instances_results.json':'12001a6dad31a60d0a6e6e4fa9e9f25dfca7f48faa520916503a96a7e2c6c396'}
TRACKED=['offline_tools/run_mva_arch_replication.py','offline_tools/mva_arch_replication.py',
 'offline_tools/test_mva_arch_replication.py','offline_tools/test_mva_phase2.py',
 'offline_tools/mva_full_coco_bootstrap.py','offline_tools/mva_fusion_controls.py',
 'offline_tools/compare_fusion_rules.py','offline_tools/fuse_coco_instances.py',
 'offline_tools/mva_score_floor_audit.py','offline_tools/mva_fusion_mechanisms.py',
 'offline_tools/run_mva_phase2.py','refine-logs/MVA_ARCH_REPLICATION_PROTOCOL_20261016.md',
 'refine-logs/MVA_PHASE2_PROTOCOL_20261015.md','refine-logs/MVA_SCORE_FLOOR_PROTOCOL_20261016.md',
 '.aris/compute/mva-arch-analysis-20261030.md']


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
    return h.hexdigest()


def save(value,path):
    tmp=path.with_suffix(path.suffix+'.tmp')
    tmp.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n')
    os.replace(str(tmp),str(path))


def hashes():
    for p,expected in INPUTS.items():
        if sha(ROOT/p)!=expected:raise RuntimeError('Frozen input changed: '+p)
    if sha(ENV/'bin/python')!=NATIVE_SHA:raise RuntimeError('Native Python changed')
    result={p:sha(ROOT/p) for p in TRACKED}
    result.update(INPUTS)
    result['native_python']=NATIVE_SHA
    for pattern in ['lib/python3.7/site-packages/pycocotools/_mask*.so',
                    'lib/python3.7/site-packages/numpy/core/_multiarray_umath*.so']:
        paths=list(ENV.glob(pattern))
        if len(paths)!=1:raise RuntimeError('Native binary path ambiguous: '+pattern)
        result[str(paths[0].relative_to(ROOT))]=sha(paths[0])
    return result


def duplicate():
    for line in subprocess.check_output(['ps','-eo','pid,comm,args'],universal_newlines=True).splitlines()[1:]:
        p=line.split(None,2)
        if (len(p)==3 and p[1].startswith('python') and int(p[0])!=os.getpid()
            and 'run_mva_arch_replication.py' in p[2] and '--worker' in p[2]
            and '--mode full' in p[2]):return True
    return False


def launch(args):
    folder=args.output_dir.resolve();folder.relative_to(ROOT/'output_offline')
    current=hashes()
    if not args.launch:
        print(json.dumps({'read_only':True,'output_exists':folder.exists(),
                          'cuda_visible_devices':'','source_sha256':current}));return
    if folder.exists():raise RuntimeError('New exclusive output required; no overwrites/retries')
    if shutil.disk_usage(str(ROOT)).free<10*1024**3:raise RuntimeError('Disk headroom below10GiB')
    if args.mode=='full':
        if duplicate():raise RuntimeError('Existing analysis worker; no duplicate')
        if args.audit_dir is None:raise RuntimeError('Independent CPU witness required')
        witness=args.audit_dir.resolve();witness.relative_to(ROOT/'output_offline')
        old=json.loads((witness/'deployment.json').read_text())
        end=json.loads((witness/'completion.json').read_text())
        report=json.loads((witness/'audit_report.json').read_text())
        if (old['mode']!='audit' or old['source_sha256']!=current
            or end['exit_code']!=0 or report['status']!='passed'):
            raise RuntimeError('Audit failed or source/spec/runtime changed')
    folder.mkdir(parents=True,exist_ok=False)
    cpus=sorted(os.sched_getaffinity(0))[:2]
    command=['ionice','-c','2','-n','7','nice','-n','10','taskset','-c',','.join(map(str,cpus)),
             str(ENV/'bin/python'),'-u',str(Path(__file__).resolve()),'--worker',
             '--mode',args.mode,'--output-dir',str(folder)]
    env=dict(os.environ)
    env.update(CUDA_VISIBLE_DEVICES='',PYTHONNOUSERSITE='1',PYTHONDONTWRITEBYTECODE='1',
               OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',
               NUMEXPR_NUM_THREADS='1',PATH=str(ENV/'bin')+':'+env.get('PATH',''),
               LD_LIBRARY_PATH=str(ENV/'lib')+':'+str(ENV/'lib/python3.7/site-packages/torch/lib'))
    state={'mode':args.mode,'started_unix':time.time(),'source_sha256':current,
           'command':command,'cpu_affinity':cpus,'cpu_workers':1,'threads':1,
           'gpu_disabled':True,'test_split_access':False,'status':'launching'}
    save(state,folder/'deployment.json')
    with (folder/'console.log').open('x') as log:
        p=subprocess.Popen(command,cwd=str(ROOT),env=env,stdin=subprocess.DEVNULL,
                           stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
    state.update(pid=p.pid,status='launched');save(state,folder/'deployment.json')
    print(json.dumps({'pid':p.pid,'folder':str(folder)}),flush=True)


def worker(args):
    folder=args.output_dir.resolve();code=0;error=None
    try:
        deployment=json.loads((folder/'deployment.json').read_text())
        if (Path(sys.prefix).resolve()!=ENV.resolve() or os.environ.get('CUDA_VISIBLE_DEVICES')!=''
            or deployment['source_sha256']!=hashes() or deployment['mode']!=args.mode):
            raise RuntimeError('Runtime/source/mode differs from launch record')
        sys.path.insert(0,str(ROOT))
        import numpy
        from offline_tools.mva_arch_replication import INPUT_HASHES,load_inputs,run
        if INPUT_HASHES!=INPUTS:raise RuntimeError('Input declaration mismatch')
        if args.mode=='audit':
            for script in ['offline_tools/test_mva_phase2.py','offline_tools/test_mva_arch_replication.py']:
                subprocess.check_call([sys.executable,'-u',script])
            dataset,values,counts=load_inputs()
            save({'status':'passed','tests':15,'native_prefix':sys.prefix,
                'numpy_version':numpy.__version__,'source_sha256':deployment['source_sha256'],
                'image_count':len(dataset['images']),'prediction_counts':counts,
                'gpu_disabled':True,'test_split_access':False,'new_fusion_ap_evaluated':False},
                folder/'audit_report.json')
        else:run(folder)
    except BaseException as exc:
        code=1;error=repr(exc);raise
    finally:
        save({'exit_code':code,'error':error,'mode':args.mode,'finished_unix':time.time(),
              'gpu_disabled':True,'test_split_access':False},folder/'completion.json')
        (folder/'exit_code.txt').write_text(str(code)+'\n')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mode',choices=['audit','full'],required=True)
    parser.add_argument('--output-dir',type=Path,required=True)
    parser.add_argument('--audit-dir',type=Path)
    parser.add_argument('--worker',action='store_true',help=argparse.SUPPRESS)
    parser.add_argument('--launch',action='store_true')
    args=parser.parse_args();os.chdir(str(ROOT))
    (worker if args.worker else launch)(args)


if __name__=='__main__':main()

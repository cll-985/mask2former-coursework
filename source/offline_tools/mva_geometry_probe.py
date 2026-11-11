"""GT-assisted exploratory geometry diagnosis; NOT a deployable fusion method."""
import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parent.parent
ENV=ROOT/'.runtime_gpu2_native_v2/conda'
GT='CarDD_COCO/annotations/instances_val2017.json'
EXPECTED_GT='25d445e2da4d86a14390d419d8147a1ace97f8f71f2e37cd2e7c6be79398a8d2'
BASES={
 '42':'output_offline/B00_formal_R50/20260819_133145_seed42/inference/coco_instances_results.json',
 '3407':'output_offline/B00_formal_R50/20260822_073123_seed3407/inference/coco_instances_results.json',
 '2026':'output_offline/B00_formal_R50/20260824_112238_seed2026/inference/coco_instances_results.json'}
RCNN=[
 'output_offline/R01_MVA_MaskRCNN_R50_20261017_gpu1_v1/seed42/inference/coco_instances_results.json',
 'output_offline/R01_MVA_MaskRCNN_seed3407_20261026_gpu0_resume_v3/stage/inference/coco_instances_results.json']


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
    return h.hexdigest()


def save(value,path):
    tmp=path.with_suffix(path.suffix+'.tmp')
    tmp.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n');os.replace(str(tmp),str(path))


def expansion_delta(t,fp,g,u,v):
    j=float(t)/(g+fp)
    return (u-j*v)/(g+fp+v)


def self_test():
    checks=0
    # Exhaustive six-pixel universes, all nonempty G and arbitrary endpoints.
    pop=lambda x:bin(x).count('1')
    for gmask in range(1,64):
        g=pop(gmask)
        for a in range(64):
            t=pop(a&gmask);fp=pop(a&~gmask);j=t/(g+fp)
            for b in range(64):
                d=b&~a;u=pop(d&gmask);v=pop(d&~gmask)
                union=a|b;actual=pop(union&gmask)/pop(union|gmask)-j
                assert abs(actual-expansion_delta(t,fp,g,u,v))<1e-12
                assert (u*(g+fp)>t*v)==(pop(union&gmask)*pop(a|gmask)>t*pop(union|gmask))
                checks+=1
    # Similar masks can still share precisely the same mistakes.
    assert expansion_delta(8,2,10,0,2)<0
    assert expansion_delta(8,2,10,2,0)>0
    assert expansion_delta(8,2,10,0,0)==0
    print(json.dumps({'status':'passed','exhaustive_cases':checks,'extra_cases':3}),flush=True)


def probe(folder):
    import numpy as np
    from pycocotools.coco import COCO
    from pycocotools import mask as mu
    sys.path.insert(0,str(ROOT))
    from offline_tools.compare_fusion_rules import group_results
    from offline_tools.fuse_coco_instances import greedy_matches,rle_for_compute
    if os.environ.get('CUDA_VISIBLE_DEVICES')!='':raise RuntimeError('Hide CUDA')
    if sha(ROOT/GT)!=EXPECTED_GT:raise RuntimeError('Frozen GT changed')
    coco=COCO(str(ROOT/GT))
    gtgroups={}
    for ann in coco.dataset['annotations']:
        if not ann.get('iscrowd',0):
            gtgroups.setdefault((ann['image_id'],ann['category_id']),[]).append(ann)
    pairs=[('mask2former_'+a+'_'+b,[BASES[a],BASES[b]]) for a,b in [('42','3407'),('42','2026'),('3407','2026')]]
    pairs.append(('maskrcnn_42_3407',RCNN))
    report={'status':'running','test_split_access':False,'gpu_disabled':True,'gt_sha256':EXPECTED_GT,
        'scope':'post-results exploratory GT-assisted local diagnosis,not new AP/independent validation',
        'selection':'all greedy IoU>=.5 matched pairs with mean score>=.001 BEFORE final global top100',
        'anchor':'same-class noncrowd GT maximizing IoU of classification-score-selected endpoint;GT-assisted selection bias',
        'no_causal_or_AP_guarantee':True,'pairs':{}}
    fields=['image_id','category_id','gt_id','score','endpoint_iou','union_iou','best_endpoint_iou',
            'added_tp','added_fp','margin','delta','both_endpoints_match_different_gt_at_05','endpoint_ge05']
    for label,paths in pairs:
        inputs={p:sha(ROOT/p) for p in paths}
        values=[json.loads((ROOT/p).read_text()) for p in paths]
        groups=[group_results(v) for v in values]
        rows=[];matched=0;below=0;no_gt=0;zero_anchor=0
        with (folder/(label+'.csv')).open('x',newline='') as f:
            writer=csv.DictWriter(f,fieldnames=fields);writer.writeheader()
            for key in sorted(set(groups[0])|set(groups[1])):
                left=sorted(groups[0][key],key=lambda x:-float(x['score']))
                right=sorted(groups[1][key],key=lambda x:-float(x['score']))
                matches=greedy_matches(left,right,.5)
                for i,j,_ in matches:
                    matched+=1;a,b=left[i],right[j]
                    score=(float(a['score'])+float(b['score']))/2
                    if score<.001:below+=1;continue
                    annotations=gtgroups.get(key,[])
                    if not annotations:no_gt+=1;continue
                    targets=[coco.annToRLE(x) for x in annotations]
                    ar,br=[rle_for_compute(x['segmentation']) for x in (a,b)]
                    overlaps=mu.iou([ar,br],targets,[0]*len(targets))
                    chosen=0 if float(a['score'])>=float(b['score']) else 1
                    target_index=int(np.argmax(overlaps[chosen]))
                    if overlaps[chosen,target_index]<=0:zero_anchor+=1;continue
                    endpoint=[ar,br][chosen];target=targets[target_index]
                    union=mu.merge([ar,br]);area=lambda r:int(mu.area(r))
                    t=area(mu.merge([endpoint,target],intersect=True))
                    fp=area(endpoint)-t;g=area(target)
                    u=area(mu.merge([union,target],intersect=True))-t
                    v=area(union)-area(endpoint)-u
                    old=t/(g+fp);new=(t+u)/(g+fp+v)
                    assert abs(new-old-expansion_delta(t,fp,g,u,v))<1e-10
                    gt0,gt1=int(np.argmax(overlaps[0])),int(np.argmax(overlaps[1]))
                    conflict=gt0!=gt1 and overlaps[0,gt0]>=.5 and overlaps[1,gt1]>=.5
                    row=dict(image_id=key[0],category_id=key[1],gt_id=annotations[target_index]['id'],
                        score=score,endpoint_iou=old,union_iou=new,best_endpoint_iou=float(max(overlaps[:,target_index])),
                        added_tp=u,added_fp=v,margin=u-old*v,delta=new-old,
                        both_endpoints_match_different_gt_at_05=int(conflict),endpoint_ge05=int(old>=.5))
                    writer.writerow(row);rows.append(row)
        summary={'source_sha256':inputs,'matched_before_floor':matched,'below_mean_score_floor':below,
                 'no_same_class_gt':no_gt,'zero_overlap_anchor':zero_anchor,'diagnosed':len(rows)}
        for stratum,subset in [('all_anchored',rows),('endpoint_ge05',[x for x in rows if x['endpoint_ge05']])]:
            n=len(subset)
            summary[stratum]={'n':n,'mean_local_iou_delta':float(np.mean([x['delta'] for x in subset])) if n else None,
                'improved':sum(x['delta']>1e-10 for x in subset),'harmed':sum(x['delta']< -1e-10 for x in subset),
                'unchanged':sum(abs(x['delta'])<=1e-10 for x in subset),
                'worse_than_best_endpoint':sum(x['union_iou']<x['best_endpoint_iou']-1e-10 for x in subset),
                'known_gt_identity_conflicts':sum(x['both_endpoints_match_different_gt_at_05'] for x in subset)}
        if any(sha(ROOT/p)!=h for p,h in inputs.items()):raise RuntimeError('Input changed during diagnosis')
        report['pairs'][label]=summary;save(report,folder/'report.json')
        print('PAIR_COMPLETE '+label+' '+json.dumps(summary),flush=True)
    report['status']='complete';save(report,folder/'report.json')


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--self-test',action='store_true');p.add_argument('--launch',action='store_true')
    p.add_argument('--output-dir',type=Path);p.add_argument('--worker',action='store_true')
    a=p.parse_args();os.chdir(str(ROOT))
    if a.self_test:self_test();return
    folder=a.output_dir.resolve();folder.relative_to(ROOT/'output_offline')
    if a.launch:
        if folder.exists():raise RuntimeError('New output required')
        folder.mkdir(parents=True)
        cpus=sorted(os.sched_getaffinity(0))[:2];env=dict(os.environ)
        env.update(CUDA_VISIBLE_DEVICES='',PYTHONNOUSERSITE='1',PYTHONDONTWRITEBYTECODE='1',
            OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1',
            LD_LIBRARY_PATH=str(ENV/'lib')+':'+str(ENV/'lib/python3.7/site-packages/torch/lib'))
        manifest={'script_sha256':sha(__file__),'cpu_affinity':cpus,'test_split_access':False,'gpu_disabled':True}
        save(manifest,folder/'deployment.json')
        cmd=['ionice','-c','2','-n','7','nice','-n','10','taskset','-c',','.join(map(str,cpus)),
             str(ENV/'bin/python'),'-u',__file__,'--worker','--output-dir',str(folder)]
        with (folder/'console.log').open('x') as log:
            child=subprocess.Popen(cmd,cwd=str(ROOT),env=env,stdin=subprocess.DEVNULL,
                stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        manifest['pid']=child.pid;save(manifest,folder/'deployment.json');print(json.dumps(manifest));return
    if not a.worker:raise RuntimeError('Use --self-test or --launch')
    code=0;error=None
    try:
        manifest=json.loads((folder/'deployment.json').read_text())
        if Path(sys.prefix).resolve()!=ENV.resolve() or sha(__file__)!=manifest['script_sha256']:
            raise RuntimeError('Runtime/source mismatch')
        probe(folder)
    except BaseException as exc:code=1;error=repr(exc);raise
    finally:save(dict(exit_code=code,error=error,finished_unix=time.time()),folder/'completion.json')


if __name__=='__main__':main()

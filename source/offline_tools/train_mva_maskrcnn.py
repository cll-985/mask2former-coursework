#!/usr/bin/env python3
"""Frozen Mask R-CNN replication; train/val only, no test registration."""
import argparse
import hashlib
import json
import logging
import math
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
OVERLAY = ROOT / '.runtime/mva_tv0100cu111'
if not (OVERLAY / 'runtime_probe.json').is_file():
    raise RuntimeError('Private torchvision runtime not prepared; see .aris/compute/remote-maskrcnn.md')
sys.path.insert(0, str(OVERLAY))

import numpy as np
import torch
import torchvision
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.config import get_cfg
from detectron2.data import (DatasetCatalog, DatasetMapper,
                            build_detection_train_loader)
from detectron2.data import transforms as T
from detectron2.data.datasets import register_coco_instances
from detectron2.engine import DefaultTrainer, default_setup, hooks
from detectron2.evaluation import COCOEvaluator
from detectron2.evaluation.coco_evaluation import instances_to_coco_json
from detectron2.modeling import build_model
from detectron2.utils.env import seed_all_rng
from detectron2.utils.events import EventStorage

CONFIG = 'configs/cardd/mva_mask_rcnn_R50_fpn_replication.yaml'
INPUTS = {
    'datasets/pretrained/R-50.pkl':
        '16fd03f0b0123e85b959cb9b64aaf9eeb4a7dc68bd9b0b134a9220d4b24c15c6',
    'CarDD_COCO/annotations/instances_train2017.json':
        '175052757f0c85a5fe7bf3d0f3d3b27fcfd5e2d7ed0246309f5ea1864ee9637a',
    'CarDD_COCO/annotations/instances_val2017.json':
        '25d445e2da4d86a14390d419d8147a1ace97f8f71f2e37cd2e7c6be79398a8d2',
}


def sha256(path):
    digest = hashlib.sha256()
    with open(path, 'rb') as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def save_json(value, path):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    os.replace(str(temporary), str(path))


def register_data():
    categories = None
    for split, count in [('train', 2816), ('val', 810)]:
        annotation = 'CarDD_COCO/annotations/instances_{}2017.json'.format(split)
        with open(annotation) as handle:
            raw = json.load(handle)
        assert len(raw['images']) == count and len(raw['categories']) == 6
        current = sorted((x['id'], x['name']) for x in raw['categories'])
        if categories is not None:
            assert categories == current, 'Train/val category mapping differs'
        categories = current
        register_coco_instances('mva_maskrcnn_cardd_' + split, {}, annotation,
                                'CarDD_COCO/{}2017'.format(split))


def make_cfg(args):
    cfg = get_cfg()
    cfg.INPUT.IMAGE_SIZE = 1024
    cfg.INPUT.MIN_SCALE = 0.1
    cfg.INPUT.MAX_SCALE = 2.0
    cfg.merge_from_file(str(ROOT / CONFIG))
    cfg.SEED = args.seed
    cfg.OUTPUT_DIR = str(args.output_dir)
    if args.smoke_only:
        # CPU witness is not a GPU/full-resolution throughput or accuracy test.
        cfg.MODEL.DEVICE = args.smoke_device
        cfg.SOLVER.AMP.ENABLED = False
        cfg.INPUT.IMAGE_SIZE = 256 if args.smoke_device == 'cpu' else 1024
        cfg.DATALOADER.NUM_WORKERS = 0
    cfg.freeze()
    return cfg


def mapper(cfg):
    # Native Mask R-CNN needs BitMasks.crop_and_resize, not a bare mask tensor.
    augmentations = [T.RandomFlip(horizontal=True, vertical=False),
                     T.ResizeScale(min_scale=cfg.INPUT.MIN_SCALE,
                                   max_scale=cfg.INPUT.MAX_SCALE,
                                   target_height=cfg.INPUT.IMAGE_SIZE,
                                   target_width=cfg.INPUT.IMAGE_SIZE),
                     T.FixedSizeCrop(crop_size=(cfg.INPUT.IMAGE_SIZE,) * 2)]
    return DatasetMapper(True, augmentations=augmentations,
                         image_format=cfg.INPUT.FORMAT, use_instance_mask=True,
                         instance_mask_format='bitmask', recompute_boxes=True)


class ReplicationTrainer(DefaultTrainer):
    @classmethod
    def build_train_loader(cls, cfg):
        return build_detection_train_loader(cfg, mapper=mapper(cfg))

    @classmethod
    def build_evaluator(cls, cfg, dataset_name, output_folder=None):
        return COCOEvaluator(dataset_name, tasks=('bbox', 'segm'),
                             distributed=False, output_dir=output_folder or
                             str(Path(cfg.OUTPUT_DIR) / 'inference'))

    def build_hooks(self):
        output = super().build_hooks()
        for index, hook in enumerate(output):
            if isinstance(hook, hooks.PeriodicCheckpointer):
                output[index] = hooks.PeriodicCheckpointer(
                    self.checkpointer, self.cfg.SOLVER.CHECKPOINT_PERIOD,
                    max_to_keep=2)
        return output


def load_verified_pretrain(model):
    DetectionCheckpointer(model).load(str(ROOT / 'datasets/pretrained/R-50.pkl'))
    # An actual equality witness catches silent backbone name/normalization mismatch.
    import pickle
    with open(ROOT / 'datasets/pretrained/R-50.pkl', 'rb') as handle:
        source = pickle.load(handle, encoding='latin1')['model']
    key = 'backbone.bottom_up.stem.conv1.weight'
    expected = source['stem.conv1.weight']
    actual = model.state_dict()[key].detach().cpu().numpy()
    assert np.array_equal(actual, expected), 'ImageNet backbone did not load'


def smoke(cfg, output):
    torch.set_num_threads(2)
    seed_all_rng(cfg.SEED)
    model = build_model(cfg)
    load_verified_pretrain(model)
    optimizer = ReplicationTrainer.build_optimizer(cfg, model)
    loader = iter(ReplicationTrainer.build_train_loader(cfg))
    losses_seen, batches = [], []
    initial = model.roi_heads.box_predictor.cls_score.weight.detach().clone()
    model.train()
    for step in range(2):
        batch = next(loader)
        assert all(hasattr(x['instances'].gt_masks, 'crop_and_resize') for x in batch)
        batches.append([int(x['image_id']) for x in batch])
        optimizer.zero_grad()
        with EventStorage(step):
            loss_dict = model(batch)
            loss = sum(loss_dict.values())
            assert torch.isfinite(loss).item(), 'Non-finite loss'
            loss.backward()
        assert all(p.grad is None or torch.isfinite(p.grad).all().item()
                   for p in model.parameters()), 'Non-finite gradient'
        optimizer.step()
        losses_seen.append({k: float(v.detach()) for k, v in loss_dict.items()})
        print('KERNEL_STEP {} {}'.format(step, losses_seen[-1]), flush=True)
    assert not torch.equal(initial, model.roi_heads.box_predictor.cls_score.weight)
    model.eval()
    with torch.no_grad():
        before = model(batch)[0]['instances'].to('cpu')
    DetectionCheckpointer(model, save_dir=str(output)).save('smoke_checkpoint')
    restored = build_model(cfg).eval()
    DetectionCheckpointer(restored).load(str(output / 'smoke_checkpoint.pth'))
    with torch.no_grad():
        after = restored(batch)[0]['instances'].to('cpu')
    assert torch.equal(before.pred_masks, after.pred_masks)
    assert torch.equal(before.pred_classes, after.pred_classes)
    assert torch.equal(before.scores, after.scores)
    save_json(instances_to_coco_json(after, int(batch[0]['image_id'])),
              output / 'train_image_smoke_predictions.json')
    return {'status': 'passed', 'device': cfg.MODEL.DEVICE, 'seed': cfg.SEED,
            'training_steps': 2, 'image_size': cfg.INPUT.IMAGE_SIZE, 'batch_size': 2,
            'batch_image_ids': batches, 'losses': losses_seen,
            'pretrained_stem_exact_match': True,
            'finite_gradients': True, 'head_weights_updated': True,
            'checkpoint_reload_predictions_exact': True,
            'gpu_kernel_tested': cfg.MODEL.DEVICE == 'cuda',
            'full_resolution_tested': cfg.INPUT.IMAGE_SIZE == 1024,
            'validation_accuracy_evaluated': False, 'test_split_access': False}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--seed', type=int, choices=(42, 3407), required=True)
    parser.add_argument('--smoke-only', action='store_true')
    parser.add_argument('--smoke-device', choices=('cpu', 'cuda'), default='cpu')
    args = parser.parse_args()
    os.chdir(str(ROOT))
    output = args.output_dir.resolve()
    output.relative_to(ROOT / 'output_offline')
    output.mkdir(parents=True, exist_ok=False)
    args.output_dir = output
    cfg = make_cfg(args)
    default_setup(cfg, args)
    assert torchvision.__version__ == '0.10.0+cu111'
    Path(torchvision.__file__).resolve().relative_to(OVERLAY)
    runtime = json.loads((OVERLAY / 'runtime_probe.json').read_text())
    assert runtime['binary_sha256'] == sha256(OVERLAY / 'torchvision/_C.so')
    for path, expected in INPUTS.items():
        assert sha256(ROOT / path) == expected, 'Frozen input changed: ' + path
    register_data()
    manifest = {'seed': args.seed, 'architecture': 'Mask R-CNN R50-FPN',
                'source_sha256': {p: sha256(ROOT / p) for p in
                    [CONFIG, 'offline_tools/train_mva_maskrcnn.py',
                     'refine-logs/MVA_ARCH_REPLICATION_PROTOCOL_20261016.md',
                     '.aris/compute/remote-maskrcnn.md']},
                'input_sha256': INPUTS, 'test_split_access': False,
                'mode': ('smoke_' + args.smoke_device) if args.smoke_only else 'formal_training',
                'runtime': runtime,
                'optimizer': 'native Detectron2 SGD', 'checkpoint_selection': 'final',
                'torch_version': torch.__version__, 'pid': os.getpid()}
    save_json(manifest, output / 'launch_manifest.json')
    start = time.monotonic()
    try:
        if args.smoke_only:
            result = smoke(cfg, output)
            save_json(result, output / 'smoke_report.json')
        else:
            if not torch.cuda.is_available():
                raise RuntimeError('CUDA unavailable; refusing CPU formal training')
            trainer = ReplicationTrainer(cfg)
            load_verified_pretrain(trainer.model)
            # Keep fresh optimizer/scheduler; no implicit checkpoint resume.
            trainer.train()
        (output / 'exit_code.txt').write_text('0\n')
        print('MVA_MASKRCNN_COMPLETE seconds={:.1f}'.format(time.monotonic()-start), flush=True)
    except Exception:
        (output / 'exit_code.txt').write_text('1\n')
        raise


if __name__ == '__main__':
    main()

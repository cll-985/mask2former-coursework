# Copyright (c) Facebook, Inc. and its affiliates.
# Modified by Bowen Cheng from https://github.com/facebookresearch/detr/blob/master/models/detr.py
"""
MaskFormer criterion.
"""
import logging

import math

import torch
import torch.nn.functional as F
from torch import nn

from detectron2.utils.comm import get_world_size
from detectron2.utils.events import get_event_storage
from detectron2.projects.point_rend.point_features import (
    get_uncertain_point_coords_with_randomness,
    point_sample,
)

from ..utils.misc import is_dist_avail_and_initialized, nested_tensor_from_tensor_list


def dice_loss(
        inputs: torch.Tensor,
        targets: torch.Tensor,
        num_masks: float,
    ):
    """
    Compute the DICE loss, similar to generalized IOU for masks
    Args:
        inputs: A float tensor of arbitrary shape.
                The predictions for each example.
        targets: A float tensor with the same shape as inputs. Stores the binary
                 classification label for each element in inputs
                (0 for the negative class and 1 for the positive class).
    """
    inputs = inputs.sigmoid()
    inputs = inputs.flatten(1)
    numerator = 2 * (inputs * targets).sum(-1)
    denominator = inputs.sum(-1) + targets.sum(-1)
    loss = 1 - (numerator + 1) / (denominator + 1)
    return loss.sum() / num_masks


dice_loss_jit = torch.jit.script(
    dice_loss
)  # type: torch.jit.ScriptModule


def sigmoid_ce_loss(
        inputs: torch.Tensor,
        targets: torch.Tensor,
        num_masks: float,
    ):
    """
    Args:
        inputs: A float tensor of arbitrary shape.
                The predictions for each example.
        targets: A float tensor with the same shape as inputs. Stores the binary
                 classification label for each element in inputs
                (0 for the negative class and 1 for the positive class).
    Returns:
        Loss tensor
    """
    loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction="none")

    return loss.mean(1).sum() / num_masks


sigmoid_ce_loss_jit = torch.jit.script(
    sigmoid_ce_loss
)  # type: torch.jit.ScriptModule


def calculate_uncertainty(logits):
    """
    We estimate uncerainty as L1 distance between 0.0 and the logit prediction in 'logits' for the
        foreground class in `classes`.
    Args:
        logits (Tensor): A tensor of shape (R, 1, ...) for class-specific or
            class-agnostic, where R is the total number of predicted masks in all images and C is
            the number of foreground classes. The values are logits.
    Returns:
        scores (Tensor): A tensor of shape (R, 1, ...) that contains uncertainty scores with
            the most uncertain locations having the highest uncertainty score.
    """
    assert logits.shape[1] == 1
    gt_class_logits = logits.clone()
    return -(torch.abs(gt_class_logits))


def morphological_boundary(masks, radius):
    """Return a differentiable morphological-gradient boundary map."""
    if radius < 1:
        raise ValueError("boundary radius must be >= 1")
    kernel_size = 2 * radius + 1
    dilated = F.max_pool2d(masks, kernel_size, stride=1, padding=radius)
    eroded = -F.max_pool2d(-masks, kernel_size, stride=1, padding=radius)
    return (dilated - eroded).clamp(0.0, 1.0)


def geometry_weights(
    target_masks,
    small_area_ratio,
    max_weight,
    min_weight=1.0,
    normalize=False,
    eps=1e-6,
):
    """Weight small and low-compactness masks without dataset-specific labels."""
    if min_weight < 0.0:
        raise ValueError("min_weight must be >= 0")
    if max_weight < min_weight:
        raise ValueError("max_weight must be >= min_weight")
    if small_area_ratio <= 0.0:
        raise ValueError("small_area_ratio must be > 0")

    binary_targets = (target_masks > 0.5).float()
    area = binary_targets.flatten(1).sum(1)
    image_area = float(target_masks.shape[-2] * target_masks.shape[-1])
    area_ratio = area / max(image_area, 1.0)
    smallness = (1.0 - area_ratio / small_area_ratio).clamp(0.0, 1.0)

    one_pixel_boundary = morphological_boundary(binary_targets, radius=1)
    perimeter = one_pixel_boundary.flatten(1).sum(1).clamp_min(1.0)
    compactness = (4.0 * math.pi * area / (perimeter.square() + eps)).clamp(0.0, 1.0)
    complexity = 1.0 - compactness

    difficulty = 0.5 * smallness + 0.5 * complexity
    weights = min_weight + (max_weight - min_weight) * difficulty
    if normalize and weights.numel() > 0:
        # Keep the average boundary-loss scale invariant to the geometry mix
        # in each batch while redistributing supervision across instances.
        weights = weights / weights.mean().clamp_min(eps)
    return weights


def geometry_weighted_boundary_loss(
    src_logits,
    target_masks,
    num_masks,
    band_width,
    dice_weight,
    bce_weight,
    small_area_ratio,
    geo_weight_max,
    geo_weight_min=1.0,
    normalize_weights=False,
):
    """Boundary Dice/BCE averaged with per-instance geometry weights."""
    if src_logits.numel() == 0:
        return src_logits.sum() * 0.0

    target_masks = F.interpolate(
        target_masks.float(),
        size=src_logits.shape[-2:],
        mode="nearest",
    )
    target_masks = (target_masks > 0.5).float()
    pred_boundary = morphological_boundary(src_logits.sigmoid(), band_width)
    target_boundary = morphological_boundary(target_masks, band_width)

    pred_flat = pred_boundary.flatten(1)
    target_flat = target_boundary.flatten(1)
    numerator = 2.0 * (pred_flat * target_flat).sum(1)
    denominator = pred_flat.sum(1) + target_flat.sum(1)
    boundary_dice = 1.0 - (numerator + 1.0) / (denominator + 1.0)

    # PyTorch 1.9 rejects probability-space BCE under AMP autocast. Boundary
    # morphology already operates on probabilities, so only this sensitive
    # term is evaluated in FP32 while retaining its gradient path.
    with torch.cuda.amp.autocast(enabled=False):
        boundary_bce = F.binary_cross_entropy(
            pred_flat.float().clamp(1e-6, 1.0 - 1e-6),
            target_flat.float(),
            reduction="none",
        ).mean(1)

    weights = geometry_weights(
        target_masks,
        small_area_ratio=small_area_ratio,
        max_weight=geo_weight_max,
        min_weight=geo_weight_min,
        normalize=normalize_weights,
    )
    per_instance = dice_weight * boundary_dice + bce_weight * boundary_bce
    return (weights * per_instance).sum() / num_masks


def soft_high_resolution_boundary(target_masks, output_size, radius):
    """Create the GT boundary before area downsampling to retain thin masks."""
    if target_masks.numel() == 0:
        return target_masks.new_zeros(
            (target_masks.shape[0], target_masks.shape[1]) + tuple(output_size)
        )
    with torch.no_grad():
        binary_targets = (target_masks.float() > 0.5).float()
        high_resolution_boundary = morphological_boundary(binary_targets, radius)
        return F.interpolate(
            high_resolution_boundary,
            size=output_size,
            mode="area",
        ).clamp_(0.0, 1.0)


def project_boundary_gradients(region_grad, boundary_grad, norm_cap=0.10, eps=1e-6):
    """Project conflicting per-instance gradients and cap their relative norm."""
    if norm_cap < 0.0:
        raise ValueError("CABR gradient norm cap must be >= 0")
    if region_grad.shape != boundary_grad.shape:
        raise ValueError("CABR region and boundary gradients must have equal shapes")
    if region_grad.numel() == 0:
        empty = region_grad.new_zeros((0,))
        return boundary_grad, {
            "conflict": empty,
            "capped": empty,
            "post_alignment": empty,
        }

    region_flat = region_grad.float().flatten(1)
    boundary_flat = boundary_grad.float().flatten(1)
    dot = (region_flat * boundary_flat).sum(1, keepdim=True)
    region_norm_sq = region_flat.square().sum(1, keepdim=True)
    conflict = dot < 0.0

    projection = torch.where(
        conflict,
        dot / (region_norm_sq + eps),
        torch.zeros_like(dot),
    )
    safe_flat = boundary_flat - projection * region_flat

    region_norm = region_norm_sq.sqrt()
    safe_norm = safe_flat.square().sum(1, keepdim=True).sqrt()
    scale = torch.minimum(
        torch.ones_like(safe_norm),
        norm_cap * region_norm / (safe_norm + eps),
    )
    capped = scale < (1.0 - 1e-7)
    safe_flat = safe_flat * scale
    post_alignment = (region_flat * safe_flat).sum(1)

    diagnostics = {
        "conflict": conflict.flatten().float().detach(),
        "capped": capped.flatten().float().detach(),
        "post_alignment": post_alignment.detach(),
    }
    return safe_flat.reshape_as(boundary_grad).to(boundary_grad), diagnostics


def conflict_aware_boundary_loss(
    src_logits,
    target_masks,
    num_masks,
    band_width=1,
    dice_weight=1.0,
    bce_weight=0.5,
    grad_norm_cap=0.10,
    eps=1e-6,
):
    """Return a boundary loss whose backward gradient cannot oppose region loss.

    The displayed value is the ordinary boundary loss. Its backward path is a
    first-order surrogate carrying the projected and norm-capped gradient.
    """
    if src_logits.numel() == 0:
        empty = src_logits.new_zeros((0,))
        return src_logits.sum() * 0.0, {
            "conflict": empty,
            "capped": empty,
            "post_alignment": empty,
        }

    logits = src_logits.float()
    with torch.no_grad():
        region_target = F.interpolate(
            target_masks.float(),
            size=logits.shape[-2:],
            mode="nearest",
        )
        region_target = (region_target > 0.5).float()
        boundary_target = soft_high_resolution_boundary(
            target_masks,
            output_size=logits.shape[-2:],
            radius=band_width,
        )

    probability = logits.sigmoid()
    probability_flat = probability.flatten(1)
    region_flat = region_target.flatten(1)
    region_bce = F.binary_cross_entropy_with_logits(
        logits,
        region_target,
        reduction="none",
    ).flatten(1).mean(1)
    region_dice = 1.0 - (
        2.0 * (probability_flat * region_flat).sum(1) + 1.0
    ) / (probability_flat.sum(1) + region_flat.sum(1) + 1.0)
    region_per_instance = region_bce + region_dice

    predicted_boundary = morphological_boundary(probability, band_width)
    predicted_flat = predicted_boundary.flatten(1)
    boundary_flat = boundary_target.flatten(1)
    boundary_dice = 1.0 - (
        2.0 * (predicted_flat * boundary_flat).sum(1) + 1.0
    ) / (predicted_flat.sum(1) + boundary_flat.sum(1) + 1.0)
    # Probability-space BCE is explicitly disallowed by PyTorch 1.9 under
    # AMP, even when its inputs were cast to FP32. Keep only this numerically
    # sensitive term outside autocast while preserving the gradient path.
    with torch.cuda.amp.autocast(enabled=False):
        boundary_bce = F.binary_cross_entropy(
            predicted_flat.float().clamp(1e-6, 1.0 - 1e-6),
            boundary_flat.float(),
            reduction="none",
        ).mean(1)
    boundary_per_instance = dice_weight * boundary_dice + bce_weight * boundary_bce

    region_grad = torch.autograd.grad(
        region_per_instance.sum(),
        src_logits,
        retain_graph=True,
        create_graph=False,
    )[0].detach()
    boundary_grad = torch.autograd.grad(
        boundary_per_instance.sum(),
        src_logits,
        retain_graph=True,
        create_graph=False,
    )[0].detach()
    safe_grad, diagnostics = project_boundary_gradients(
        region_grad,
        boundary_grad,
        norm_cap=grad_norm_cap,
        eps=eps,
    )

    gradient_surrogate = (src_logits * safe_grad.to(src_logits)).sum() / num_masks
    reported_loss = boundary_per_instance.detach().sum() / num_masks
    loss = gradient_surrogate - gradient_surrogate.detach() + reported_loss
    return loss, diagnostics


class SetCriterion(nn.Module):
    """This class computes the loss for DETR.
    The process happens in two steps:
        1) we compute hungarian assignment between ground truth boxes and the outputs of the model
        2) we supervise each pair of matched ground-truth / prediction (supervise class and box)
    """

    def __init__(self, num_classes, matcher, weight_dict, eos_coef, losses,
                 num_points, oversample_ratio, importance_sample_ratio,
                 boundary_band_width=1, boundary_dice_weight=1.0,
                 boundary_bce_weight=0.5, boundary_small_area_ratio=0.01,
                 boundary_geo_weight_min=1.0, boundary_geo_weight_max=2.0,
                 boundary_normalize_weights=False, boundary_aux_layers=3,
                 cabr_band_width=1, cabr_dice_weight=1.0,
                 cabr_bce_weight=0.5, cabr_grad_norm_cap=0.10,
                 cabr_eps=1e-6):
        """Create the criterion.
        Parameters:
            num_classes: number of object categories, omitting the special no-object category
            matcher: module able to compute a matching between targets and proposals
            weight_dict: dict containing as key the names of the losses and as values their relative weight.
            eos_coef: relative classification weight applied to the no-object category
            losses: list of all the losses to be applied. See get_loss for list of available losses.
        """
        super().__init__()
        self.num_classes = num_classes
        self.matcher = matcher
        self.weight_dict = weight_dict
        self.eos_coef = eos_coef
        self.losses = losses
        empty_weight = torch.ones(self.num_classes + 1)
        empty_weight[-1] = self.eos_coef
        self.register_buffer("empty_weight", empty_weight)

        # pointwise mask loss parameters
        self.num_points = num_points
        self.oversample_ratio = oversample_ratio
        self.importance_sample_ratio = importance_sample_ratio
        self.boundary_band_width = boundary_band_width
        self.boundary_dice_weight = boundary_dice_weight
        self.boundary_bce_weight = boundary_bce_weight
        self.boundary_small_area_ratio = boundary_small_area_ratio
        self.boundary_geo_weight_min = boundary_geo_weight_min
        self.boundary_geo_weight_max = boundary_geo_weight_max
        self.boundary_normalize_weights = boundary_normalize_weights
        self.boundary_aux_layers = boundary_aux_layers
        self.cabr_band_width = cabr_band_width
        self.cabr_dice_weight = cabr_dice_weight
        self.cabr_bce_weight = cabr_bce_weight
        self.cabr_grad_norm_cap = cabr_grad_norm_cap
        self.cabr_eps = cabr_eps

    def loss_labels(self, outputs, targets, indices, num_masks):
        """Classification loss (NLL)
        targets dicts must contain the key "labels" containing a tensor of dim [nb_target_boxes]
        """
        assert "pred_logits" in outputs
        src_logits = outputs["pred_logits"].float()

        idx = self._get_src_permutation_idx(indices)
        target_classes_o = torch.cat([t["labels"][J] for t, (_, J) in zip(targets, indices)])
        target_classes = torch.full(
            src_logits.shape[:2], self.num_classes, dtype=torch.int64, device=src_logits.device
        )
        target_classes[idx] = target_classes_o

        loss_ce = F.cross_entropy(src_logits.transpose(1, 2), target_classes, self.empty_weight)
        losses = {"loss_ce": loss_ce}
        return losses
    
    def loss_masks(self, outputs, targets, indices, num_masks):
        """Compute the losses related to the masks: the focal loss and the dice loss.
        targets dicts must contain the key "masks" containing a tensor of dim [nb_target_boxes, h, w]
        """
        assert "pred_masks" in outputs

        src_idx = self._get_src_permutation_idx(indices)
        tgt_idx = self._get_tgt_permutation_idx(indices)
        src_masks = outputs["pred_masks"]
        src_masks = src_masks[src_idx]
        masks = [t["masks"] for t in targets]
        # TODO use valid to mask invalid areas due to padding in loss
        target_masks, valid = nested_tensor_from_tensor_list(masks).decompose()
        target_masks = target_masks.to(src_masks)
        target_masks = target_masks[tgt_idx]

        # No need to upsample predictions as we are using normalized coordinates :)
        # N x 1 x H x W
        src_masks = src_masks[:, None]
        target_masks = target_masks[:, None]

        with torch.no_grad():
            # sample point_coords
            point_coords = get_uncertain_point_coords_with_randomness(
                src_masks,
                lambda logits: calculate_uncertainty(logits),
                self.num_points,
                self.oversample_ratio,
                self.importance_sample_ratio,
            )
            # get gt labels
            point_labels = point_sample(
                target_masks,
                point_coords,
                align_corners=False,
            ).squeeze(1)

        point_logits = point_sample(
            src_masks,
            point_coords,
            align_corners=False,
        ).squeeze(1)

        losses = {
            "loss_mask": sigmoid_ce_loss_jit(point_logits, point_labels, num_masks),
            "loss_dice": dice_loss_jit(point_logits, point_labels, num_masks),
        }

        del src_masks
        del target_masks
        return losses

    def loss_boundaries(self, outputs, targets, indices, num_masks):
        """Geometry-weighted boundary consistency for matched masks."""
        assert "pred_masks" in outputs
        src_idx = self._get_src_permutation_idx(indices)
        tgt_idx = self._get_tgt_permutation_idx(indices)
        src_masks = outputs["pred_masks"][src_idx][:, None]

        if src_masks.numel() == 0:
            return {"loss_boundary": outputs["pred_masks"].sum() * 0.0}

        masks = [t["masks"] for t in targets]
        target_masks, _ = nested_tensor_from_tensor_list(masks).decompose()
        target_masks = target_masks.to(src_masks)[tgt_idx][:, None]

        loss_boundary = geometry_weighted_boundary_loss(
            src_masks,
            target_masks,
            num_masks=num_masks,
            band_width=self.boundary_band_width,
            dice_weight=self.boundary_dice_weight,
            bce_weight=self.boundary_bce_weight,
            small_area_ratio=self.boundary_small_area_ratio,
            geo_weight_max=self.boundary_geo_weight_max,
            geo_weight_min=self.boundary_geo_weight_min,
            normalize_weights=self.boundary_normalize_weights,
        )
        return {"loss_boundary": loss_boundary}

    def loss_cabr(self, outputs, targets, indices, num_masks):
        """Conflict-aware boundary refinement on final matched mask logits."""
        assert "pred_masks" in outputs
        src_idx = self._get_src_permutation_idx(indices)
        tgt_idx = self._get_tgt_permutation_idx(indices)
        src_masks = outputs["pred_masks"][src_idx][:, None]

        if src_masks.numel() == 0:
            return {"loss_cabr": outputs["pred_masks"].sum() * 0.0}

        masks = [t["masks"] for t in targets]
        target_masks, _ = nested_tensor_from_tensor_list(masks).decompose()
        target_masks = target_masks.to(src_masks)[tgt_idx][:, None]
        loss_cabr, diagnostics = conflict_aware_boundary_loss(
            src_masks,
            target_masks,
            num_masks=num_masks,
            band_width=self.cabr_band_width,
            dice_weight=self.cabr_dice_weight,
            bce_weight=self.cabr_bce_weight,
            grad_norm_cap=self.cabr_grad_norm_cap,
            eps=self.cabr_eps,
        )
        try:
            storage = get_event_storage()
            storage.put_scalar("cabr/conflict_rate", diagnostics["conflict"].mean())
            storage.put_scalar("cabr/cap_rate", diagnostics["capped"].mean())
            storage.put_scalar(
                "cabr/min_post_alignment",
                diagnostics["post_alignment"].min(),
            )
        except (AssertionError, RuntimeError):
            # Pure unit tests can call the loss without Detectron2 event storage.
            pass
        return {"loss_cabr": loss_cabr}

    def _get_src_permutation_idx(self, indices):
        # permute predictions following indices
        batch_idx = torch.cat([torch.full_like(src, i) for i, (src, _) in enumerate(indices)])
        src_idx = torch.cat([src for (src, _) in indices])
        return batch_idx, src_idx

    def _get_tgt_permutation_idx(self, indices):
        # permute targets following indices
        batch_idx = torch.cat([torch.full_like(tgt, i) for i, (_, tgt) in enumerate(indices)])
        tgt_idx = torch.cat([tgt for (_, tgt) in indices])
        return batch_idx, tgt_idx

    def get_loss(self, loss, outputs, targets, indices, num_masks):
        loss_map = {
            'labels': self.loss_labels,
            'masks': self.loss_masks,
            'boundaries': self.loss_boundaries,
            'cabr': self.loss_cabr,
        }
        assert loss in loss_map, f"do you really want to compute {loss} loss?"
        return loss_map[loss](outputs, targets, indices, num_masks)

    def forward(self, outputs, targets):
        """This performs the loss computation.
        Parameters:
             outputs: dict of tensors, see the output specification of the model for the format
             targets: list of dicts, such that len(targets) == batch_size.
                      The expected keys in each dict depends on the losses applied, see each loss' doc
        """
        outputs_without_aux = {k: v for k, v in outputs.items() if k != "aux_outputs"}

        # Retrieve the matching between the outputs of the last layer and the targets
        indices = self.matcher(outputs_without_aux, targets)

        # Compute the average number of target boxes accross all nodes, for normalization purposes
        num_masks = sum(len(t["labels"]) for t in targets)
        num_masks = torch.as_tensor(
            [num_masks], dtype=torch.float, device=next(iter(outputs.values())).device
        )
        if is_dist_avail_and_initialized():
            torch.distributed.all_reduce(num_masks)
        num_masks = torch.clamp(num_masks / get_world_size(), min=1).item()

        # Compute all the requested losses
        losses = {}
        for loss in self.losses:
            losses.update(self.get_loss(loss, outputs, targets, indices, num_masks))

        # In case of auxiliary losses, we repeat this process with the output of each intermediate layer.
        if "aux_outputs" in outputs:
            num_aux_outputs = len(outputs["aux_outputs"])
            for i, aux_outputs in enumerate(outputs["aux_outputs"]):
                indices = self.matcher(aux_outputs, targets)
                for loss in self.losses:
                    if loss == "cabr":
                        continue
                    if (
                        loss == "boundaries"
                        and i < num_aux_outputs - self.boundary_aux_layers
                    ):
                        continue
                    l_dict = self.get_loss(loss, aux_outputs, targets, indices, num_masks)
                    l_dict = {k + f"_{i}": v for k, v in l_dict.items()}
                    losses.update(l_dict)

        return losses

    def __repr__(self):
        head = "Criterion " + self.__class__.__name__
        body = [
            "matcher: {}".format(self.matcher.__repr__(_repr_indent=8)),
            "losses: {}".format(self.losses),
            "weight_dict: {}".format(self.weight_dict),
            "num_classes: {}".format(self.num_classes),
            "eos_coef: {}".format(self.eos_coef),
            "num_points: {}".format(self.num_points),
            "oversample_ratio: {}".format(self.oversample_ratio),
            "importance_sample_ratio: {}".format(self.importance_sample_ratio),
        ]
        _repr_indent = 4
        lines = [head] + [" " * _repr_indent + line for line in body]
        return "\n".join(lines)

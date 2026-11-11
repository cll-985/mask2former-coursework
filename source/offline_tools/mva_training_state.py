"""Explicit inner-trainer serialization for the pinned Detectron2 runtime.

Never fabricate missing legacy momentum/scaler state or call it exact resume.
This preserves optimizer/scaler/scheduler, not DataLoader prefetch/RNG position.
"""


def validate_inner_state(state, amp_required=True):
    inner = state.get('_trainer')
    if not isinstance(inner, dict):
        raise ValueError('Checkpoint lacks inner trainer; exact optimizer resume is unavailable')
    opt = inner.get('optimizer')
    if not isinstance(opt, dict) or not opt.get('param_groups'):
        raise ValueError('Checkpoint lacks optimizer parameter groups')
    if state.get('iteration', -1) >= 0 and not opt.get('state'):
        raise ValueError('Trained SGD checkpoint lacks momentum state')
    if amp_required and not inner.get('grad_scaler'):
        raise ValueError('AMP checkpoint lacks GradScaler state')
    return inner


class InnerTrainerCheckpointMixin:
    """Bridge DefaultTrainer's outer hooks to its AMPTrainer state explicitly."""

    def state_dict(self):
        state = super().state_dict()
        state['_trainer'] = self._trainer.state_dict()
        state['mva_inner_state_schema'] = 1
        return state

    def load_state_dict(self, state):
        # Validate before changing even the outer iteration/LR state.
        inner = validate_inner_state(state, bool(self.cfg.SOLVER.AMP.ENABLED))
        super().load_state_dict(state)
        self._trainer.load_state_dict(inner)


def checkpoint_summary(checkpoint):
    state = checkpoint.get('trainer', {})
    inner = state.get('_trainer', {})
    optimizer = inner.get('optimizer', {})
    result = {
        'iteration': checkpoint.get('iteration', state.get('iteration')),
        'trainer_iteration': state.get('iteration'),
        'model_present': bool(checkpoint.get('model')),
        'optimizer_state_count': len(optimizer.get('state', {})),
        'optimizer_groups': len(optimizer.get('param_groups', [])),
        'grad_scaler_present': bool(inner.get('grad_scaler')),
        'scheduler_present': bool(state.get('hooks', {}).get('LRScheduler')),
        'schema': state.get('mva_inner_state_schema'),
        'exact_data_stream_resume_claimed': False,
    }
    try:
        validate_inner_state(state, amp_required=True)
        if not result['scheduler_present']:
            raise ValueError('Scheduler hook state missing')
        if result['iteration'] != result['trainer_iteration']:
            raise ValueError('Checkpoint iteration counters differ')
        result['optimizer_resume_status'] = 'complete'
    except ValueError as exc:
        result['optimizer_resume_status'] = 'incomplete'
        result['reason'] = str(exc)
    return result

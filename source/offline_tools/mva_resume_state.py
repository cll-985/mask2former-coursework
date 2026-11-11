"""Verified optimizer-state resume, explicitly not exact RNG/DataLoader continuation."""
import hashlib


def file_sha(path):
    digest = hashlib.sha256()
    with open(path, 'rb') as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def assert_tree_equal(actual, expected, torch, label='state'):
    if torch.is_tensor(expected):
        if not torch.is_tensor(actual) or not torch.equal(actual.detach().cpu(), expected.detach().cpu()):
            raise ValueError('Reload mismatch: ' + label)
    elif isinstance(expected, dict):
        if not isinstance(actual, dict) or set(actual) != set(expected):
            raise ValueError('Reload keys mismatch: ' + label)
        for key in expected:
            assert_tree_equal(actual[key], expected[key], torch, label + '.' + str(key))
    elif isinstance(expected, (list, tuple)):
        if not isinstance(actual, (list, tuple)) or len(actual) != len(expected):
            raise ValueError('Reload sequence mismatch: ' + label)
        for i, (a, b) in enumerate(zip(actual, expected)):
            assert_tree_equal(a, b, torch, label + '.' + str(i))
    elif actual != expected:
        raise ValueError('Reload scalar mismatch: ' + label)


def load_checked(path, torch, expected_sha=None, expected_iteration=None):
    from mva_training_state import checkpoint_summary
    if expected_sha is not None and file_sha(path) != expected_sha:
        raise ValueError('Source checkpoint SHA256 changed')
    checkpoint = torch.load(str(path), map_location='cpu')
    summary = checkpoint_summary(checkpoint)
    if summary['optimizer_resume_status'] != 'complete':
        raise ValueError('Incomplete training state: ' + repr(summary))
    iteration = summary['iteration']
    if not isinstance(iteration, int) or iteration < 0:
        raise ValueError('Invalid completed-iteration counter')
    if expected_iteration is not None and iteration != expected_iteration:
        raise ValueError('Unexpected source checkpoint iteration')
    model = checkpoint['model']
    if not model or not all(bool(torch.isfinite(x).all()) for x in model.values() if torch.is_tensor(x)):
        raise ValueError('Nonfinite or empty source model')
    optimizer = checkpoint['trainer']['_trainer']['optimizer']
    if not all('momentum_buffer' in x and bool(torch.isfinite(x['momentum_buffer']).all())
               for x in optimizer['state'].values()):
        raise ValueError('Missing/nonfinite SGD momentum')
    summary.update(completed_steps=iteration + 1, model_tensor_count=len(model),
                   model_tensors_finite=True, momentum_tensors_finite=True)
    return checkpoint, summary


def restore_full_state(trainer, path, torch, expected_sha=None, expected_iteration=None):
    checkpoint, summary = load_checked(path, torch, expected_sha, expected_iteration)
    current = trainer.model.state_dict()
    if set(current) != set(checkpoint['model']):
        raise ValueError('Model keys differ; refusing partial restoration')
    if any(current[k].shape != v.shape for k, v in checkpoint['model'].items()):
        raise ValueError('Model shapes differ; refusing partial restoration')
    # Validation above precedes mutation. Native loader restores model and the
    # mixin restores outer scheduler plus native inner optimizer/AMP scaler.
    trainer.checkpointer.load(str(path))
    actual = trainer.state_dict()
    expected = checkpoint['trainer']
    assert_tree_equal(trainer.model.state_dict(), checkpoint['model'], torch, 'model')
    assert_tree_equal(actual['_trainer']['optimizer'], expected['_trainer']['optimizer'], torch, 'optimizer')
    assert_tree_equal(actual['_trainer']['grad_scaler'], expected['_trainer']['grad_scaler'], torch, 'scaler')
    assert_tree_equal(actual['hooks']['LRScheduler'], expected['hooks']['LRScheduler'], torch, 'scheduler')
    if trainer.iter != summary['iteration'] or actual['_trainer']['iteration'] != summary['iteration']:
        raise ValueError('Trainer counters differ after restoration')
    trainer.start_iter = summary['iteration'] + 1
    summary.update(start_iter=trainer.start_iter, restored_model_exact=True,
                   restored_optimizer_exact=True, restored_scaler_exact=True,
                   restored_scheduler_exact=True, rng_state_restored=False,
                   dataloader_position_restored=False,
                   lr=trainer.optimizer.param_groups[0]['lr'])
    return summary

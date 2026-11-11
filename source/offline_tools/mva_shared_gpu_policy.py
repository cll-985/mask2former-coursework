"""Small, testable cooperative-sharing policy; not hardware GPU isolation."""
from dataclasses import dataclass


@dataclass(frozen=True)
class SharingPolicy:
    memory_limit_mb: int = 3584
    reserve_mb: int = 2048
    context_allowance_mb: int = 2048
    duty_cycle: float = 0.20

    def __post_init__(self):
        if not 1024 <= self.memory_limit_mb <= 3584:
            raise ValueError('Shared allocator limit must be 1024..3584 MiB')
        if self.reserve_mb < 2048 or self.context_allowance_mb < 2048:
            raise ValueError('Insufficient external-task reserve')
        if not 0 < self.duty_cycle <= 0.20:
            raise ValueError('Duty cycle must be >0 and <=0.20')

    @property
    def start_free_mb(self):
        return self.memory_limit_mb + self.reserve_mb + self.context_allowance_mb

    def delay(self, active_seconds):
        if active_seconds < 0:
            raise ValueError('Negative elapsed time')
        return active_seconds * (1.0 / self.duty_cycle - 1.0)

    def start_allowed(self, free_mb):
        return free_mb >= self.start_free_mb

    def step_allowed(self, free_mb):
        return free_mb >= self.reserve_mb

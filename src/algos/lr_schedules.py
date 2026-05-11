"""Learning-rate schedules shared by online bandit algorithms."""

from __future__ import annotations

from dataclasses import dataclass


SUPPORTED_LR_SCHEDULES = {"constant", "shifted_inverse_time"}
DEFAULT_SHIFTED_INVERSE_TIME_OFFSET = 100


def _coalesce(value, default):
    return default if value is None else value


@dataclass
class LearningRateSchedule:
    """Stateful per-update learning-rate schedule.

    `initial_lr` is always interpreted as the first effective step size.
    For shifted inverse time, update step 1 therefore returns `initial_lr`,
    and later updates decay as:

        lr_t = initial_lr * (offset + 1) / (t + offset)
    """

    initial_lr: float
    schedule: str = "constant"
    offset: int = DEFAULT_SHIFTED_INVERSE_TIME_OFFSET
    step_count: int = 0

    def __post_init__(self) -> None:
        self.initial_lr = float(self.initial_lr)
        if self.initial_lr < 0.0:
            raise ValueError("initial_lr must be non-negative")
        self.schedule = str(self.schedule or "constant").lower()
        if self.schedule not in SUPPORTED_LR_SCHEDULES:
            raise ValueError(
                f"Unknown learning-rate schedule: {self.schedule}. "
                f"Supported: {sorted(SUPPORTED_LR_SCHEDULES)}"
            )
        self.offset = int(self.offset)
        if self.offset < 0:
            raise ValueError("learning-rate schedule offset must be non-negative")
        self.step_count = int(self.step_count)
        if self.step_count < 0:
            raise ValueError("step_count must be non-negative")

    def value(self, update_step: int) -> float:
        update_step = int(update_step)
        if update_step < 1:
            raise ValueError("update_step must be at least 1")
        if self.schedule == "constant":
            return self.initial_lr
        return self.initial_lr * float(self.offset + 1) / float(update_step + self.offset)

    def next(self) -> float:
        self.step_count += 1
        return self.value(self.step_count)

    @property
    def current_lr(self) -> float:
        if self.step_count <= 0:
            return self.initial_lr
        return self.value(self.step_count)


def build_lr_schedule(
    config: dict,
    prefix: str,
    *,
    fallback_prefix: str | None = None,
    default_lr: float = 0.01,
) -> LearningRateSchedule:
    """Build a schedule from config keys like `theta_lr_schedule`.

    `prefix_lr` remains the first effective learning rate. The optional
    fallback prefix lets nuisance schedules inherit actor defaults while still
    respecting explicit `nuisance_lr` and `nuisance_lr_schedule` overrides.
    """

    lr_key = f"{prefix}_lr"
    schedule_key = f"{prefix}_lr_schedule"
    offset_key = f"{prefix}_lr_offset"

    if fallback_prefix is None:
        initial_lr = _coalesce(config.get(lr_key), default_lr)
        schedule = _coalesce(config.get(schedule_key), "constant")
        offset = _coalesce(config.get(offset_key), DEFAULT_SHIFTED_INVERSE_TIME_OFFSET)
    else:
        fallback_lr_key = f"{fallback_prefix}_lr"
        fallback_schedule_key = f"{fallback_prefix}_lr_schedule"
        fallback_offset_key = f"{fallback_prefix}_lr_offset"
        initial_lr = _coalesce(config.get(lr_key), _coalesce(config.get(fallback_lr_key), default_lr))
        schedule = _coalesce(config.get(schedule_key), _coalesce(config.get(fallback_schedule_key), "constant"))
        offset = _coalesce(
            config.get(offset_key),
            _coalesce(config.get(fallback_offset_key), DEFAULT_SHIFTED_INVERSE_TIME_OFFSET),
        )

    return LearningRateSchedule(initial_lr=float(initial_lr), schedule=str(schedule), offset=int(offset))


def set_optimizer_lr(optimizer, lr: float) -> None:
    """Update every param-group LR for a torch optimizer-like object."""

    if optimizer is None:
        return
    for group in optimizer.param_groups:
        group["lr"] = float(lr)

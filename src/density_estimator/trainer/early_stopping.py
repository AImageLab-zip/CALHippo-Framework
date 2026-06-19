import torch
from loguru import logger


class EarlyStopping:
    """Patience-based early stopping with best-weight restoration.

    Monitors a scalar metric (e.g. validation loss).  If no improvement of
    at least ``min_delta`` is seen for ``patience`` consecutive epochs the
    ``should_stop`` flag is set and the best model weights are restored.

    Best-practice notes (from *Deep Learning*, Goodfellow et al. 2016 §7.8
    and Prechelt 2002 "Early Stopping – But When?"):

    * Always save the weights at the point of lowest validation error and
      restore them when training stops.
    * Use a generous patience — validation error curves often have several
      local minima; stopping too eagerly leaves performance on the table.
    * A small ``min_delta`` avoids counting negligible fluctuations as
      improvement.
    """

    def __init__(
        self,
        patience: int = 20,
        min_delta: float = 0.001,
        mode: str = "min",
    ) -> None:
        if mode not in ("min", "max"):
            raise ValueError(f"mode must be 'min' or 'max', got {mode!r}")
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode

        self.best_score: float | None = None
        self.best_weights: dict | None = None
        self.counter: int = 0
        self.should_stop: bool = False
        self.best_epoch: int = 0

    # ------------------------------------------------------------------

    def _is_improvement(self, current: float) -> bool:
        assert self.best_score is not None
        if self.mode == "min":
            return current < self.best_score - self.min_delta
        return current > self.best_score + self.min_delta

    def step(self, metric: float, epoch: int, model: torch.nn.Module) -> None:
        """Call once per epoch after validation.  No-op when ``patience <= 0``."""
        if self.patience <= 0:
            return  # early stopping disabled

        if self.best_score is None or self._is_improvement(metric):
            self.best_score = metric
            self.best_epoch = epoch
            self.counter = 0
            # Deep-copy weights (CPU to avoid extra GPU memory)
            self.best_weights = {
                k: v.detach().cpu().clone() for k, v in model.state_dict().items()
            }
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.should_stop = True

    def restore_best_weights(self, model: torch.nn.Module) -> None:
        """Load the saved best weights back into *model*."""
        if self.best_weights is not None:
            model.load_state_dict(self.best_weights)
            logger.info(
                f"Early-stopping: restored best weights from epoch {self.best_epoch}"
            )

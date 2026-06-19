import os

import torch
import torch.nn.functional as F
from loguru import logger


def get_dataloader_num_workers(num_loaders: int = 1) -> int:
    """Determine the optimal ``num_workers`` for :class:`DataLoader`.

    Resolution order:

    1. ``SLURM_CPUS_PER_TASK`` — most common for single-task SLURM jobs.
    2. ``SLURM_CPUS_ON_NODE`` — fallback for multi-node SLURM jobs.
    3. ``os.cpu_count()``     — local machine fallback.

    **Heuristics** (following PyTorch best practices):

    * Reserve **2 CPUs** for the main process (training loop, gradient
      computation, logging, etc.).
    * Divide the remaining CPUs evenly across the *num_loaders* that
      will run concurrently (e.g. train + val during CV = 2).
    * Cap each loader at **8 workers** — beyond that the memory overhead
      from worker-process cloning typically outweighs the I/O gains
      (each worker replicates the parent process' Python objects).
    * Floor at **0** (main-process loading) when very few CPUs are
      available, since spawning workers on a single core just adds
      overhead.

    Args:
        num_loaders: Number of DataLoaders that will be active
            simultaneously so we can split the budget.

    Returns:
        Recommended ``num_workers`` per DataLoader.
    """
    if "SLURM_CPUS_PER_TASK" in os.environ:
        total_cpus = int(os.environ["SLURM_CPUS_PER_TASK"])
    elif "SLURM_CPUS_ON_NODE" in os.environ:
        total_cpus = int(os.environ["SLURM_CPUS_ON_NODE"])
    else:
        total_cpus = os.cpu_count() or 1

    # Reserve 2 CPUs for the main process, split the rest
    available = max(total_cpus - 2, 0)
    per_loader = available // max(num_loaders, 1)

    # Cap per-loader at 8 to limit memory overhead
    workers = min(per_loader, 8)

    logger.debug(
        f"DataLoader workers: {workers}/loader  "
        f"(total_cpus={total_cpus}, reserved=2, "
        f"num_loaders={num_loaders})"
    )
    return workers


# ---------------------------------------------------------------------------
# Patched-density helpers
# ---------------------------------------------------------------------------


def patchify(
    masks: torch.Tensor,
    patch_size: int,
    use_log: bool = False,
) -> torch.Tensor:
    """Patchify density maps, optionally applying ``log1p``.

    ``avg_pool2d`` with *kernel_size=patch_size* computes the spatial mean;
    multiplying by ``patch_size²`` converts it to the **sum** of counts
    inside each patch.  When *use_log* is ``True``, ``log1p`` is then
    applied to compress dynamic range.

    Args:
        masks: ``(B, C, H, W)`` raw density maps.
        patch_size: Spatial patch side (e.g. 4 → 4×4 patches).
        use_log: If ``True``, apply ``log1p`` after patchifying.

    Returns:
        ``(B, C, H/p, W/p)`` patched (and optionally log-compressed) maps.
    """
    patched = F.avg_pool2d(masks, kernel_size=patch_size) * (patch_size**2)
    if use_log:
        patched = torch.log1p(patched)
    return patched


def to_counts(
    preds: torch.Tensor,
    use_log: bool = False,
) -> torch.Tensor:
    """Convert model predictions back to per-patch counts.

    When *use_log* is ``True``, applies ``expm1`` (inverse of ``log1p``)
    and clamps to ≥ 0.  Otherwise returns the predictions clamped to ≥ 0.
    """

    # FIXME: prima su evaluate c'era anche un campo patch_size_out
    # bisona usarlo in questa funzione?

    if use_log:
        return torch.clamp(torch.expm1(preds), min=0.0)
    return torch.clamp(preds, min=0.0)

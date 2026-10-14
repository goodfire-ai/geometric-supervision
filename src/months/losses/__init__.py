"""Training losses: task cross-entropy and the geometry losses."""
from .ce import cross_entropy_on_answer
from .geometry import geometry_loss, class_centroids, within_class_spread, ideal_ring
from .variants import LOSS_FUNCTIONS, per_prompt_loss, per_prompt_loss_with_spread

__all__ = [
    "cross_entropy_on_answer",
    "geometry_loss", "class_centroids", "within_class_spread", "ideal_ring",
    "LOSS_FUNCTIONS", "per_prompt_loss", "per_prompt_loss_with_spread",
]

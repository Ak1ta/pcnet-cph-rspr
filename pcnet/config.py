from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Dict, Tuple


ATTRIBUTE_NAMES: Tuple[str, ...] = (
    "BSO", "BW", "CB", "CIB", "CSO", "IC", "LI", "MSO",
    "OF", "SA", "SL", "SSO", "TC", "TI", "TM",
)

EXPECTED_TRAIN_SAMPLES = 10_000
EXPECTED_TEST_SAMPLES = 10_000
REFERENCE_EPOCHS = 80
REFERENCE_BATCH_SIZE = 4
REFERENCE_IMAGE_SIZE = 384
REFERENCE_LEARNING_RATE = 1e-5
REFERENCE_WEIGHT_DECAY = 1e-4
REFERENCE_CLIP = 0.5
REFERENCE_SEED = 42
SELECTION_START_EPOCH = 60
PROTOCOL_VERSION = 6


@dataclass(frozen=True)
class ExperimentSpec:
    key: str
    label: str
    challenge_perception: bool
    rspr: bool


EXPERIMENT_ORDER: Tuple[str, ...] = (
    "pcnet",
    "cph",
    "rspr",
    "cph_rspr",
)

EXPERIMENTS: Dict[str, ExperimentSpec] = {
    "pcnet": ExperimentSpec("pcnet", "PCNet", False, False),
    "cph": ExperimentSpec("cph", "PCNet + CPH", True, False),
    "rspr": ExperimentSpec("rspr", "PCNet + RSPR", False, True),
    "cph_rspr": ExperimentSpec(
        "cph_rspr", "PCNet + CPH + RSPR", True, True
    ),
}


def get_experiment(key: str) -> ExperimentSpec:
    try:
        return EXPERIMENTS[key]
    except KeyError as error:
        raise ValueError(
            "Unknown experiment {!r}; expected one of {}.".format(
                key, ", ".join(EXPERIMENT_ORDER)
            )
        ) from error


def selection_epochs(total_epochs: int) -> Tuple[int, ...]:
    """Released-PCNet-compatible Test selection schedule."""

    if total_epochs < 1:
        raise ValueError("total_epochs must be positive.")
    epochs = {1}
    epochs.update(
        range(SELECTION_START_EPOCH, min(total_epochs, REFERENCE_EPOCHS) + 1)
    )
    return tuple(sorted(epochs))


@dataclass
class ModelConfig:
    experiment: str
    seed: int = REFERENCE_SEED
    num_attributes: int = len(ATTRIBUTE_NAMES)
    attribute_hidden_dim: int = 256
    dropout: float = 0.1
    rspr_channels: int = 64
    rspr_grid_size: int = 2
    rspr_confidence_gamma: float = 4.0
    rspr_epsilon: float = 1e-6
    activation_checkpoint: bool = False
    gpu_id: int = 0
    mixed_precision: bool = False
    ihn_lev0: bool = True
    ihn_lev1: bool = False
    ihn_weight: bool = False

    def __post_init__(self) -> None:
        get_experiment(self.experiment)
        if self.num_attributes != len(ATTRIBUTE_NAMES):
            raise ValueError(
                "UVT20K CPH requires exactly {} attributes.".format(
                    len(ATTRIBUTE_NAMES)
                )
            )
        if self.rspr_channels < 8 or self.rspr_grid_size < 1:
            raise ValueError("Invalid RSPR channel/grid configuration.")
        if self.rspr_confidence_gamma <= 0 or self.rspr_epsilon <= 0:
            raise ValueError("RSPR gamma and epsilon must be positive.")

    @property
    def spec(self) -> ExperimentSpec:
        return get_experiment(self.experiment)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, values: Dict[str, Any]) -> "ModelConfig":
        allowed = set(cls.__dataclass_fields__)
        return cls(**{key: value for key, value in values.items() if key in allowed})


class IHNConfig:
    """Minimal namespace expected by the released IHN code."""

    def __init__(self, config: ModelConfig):
        self.gpuid = [config.gpu_id]
        self.mixed_precision = config.mixed_precision
        self.lev0 = config.ihn_lev0
        self.lev1 = config.ihn_lev1
        self.weight = config.ihn_weight

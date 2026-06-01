from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Union

from peft.config import PeftConfig
from peft.utils import PeftType

@dataclass
class GenFTConfig(PeftConfig):

    generator_share_dim: int = field(
        default=90,
    )
    individual_features: int = field(
        default=10,
    )
    individual_init_a: str = field(
        default='zero',
    )
    individual_init_b: str = field(
        default='zero',
    )
    ratio_W0: float = field(
        default=1.0,
    )
    inner_activation: str = field(
        default='None',
    )
    outer_activation: str = field(
        default='None',
    )
    scaling: float = field(
        default=1.0,
    )

    fan_in_fan_out: bool = field(
        default=False,
    )

    bias: str = field(
        default="none",
    )

    drop: float = field(
        default=0.0,
    )

    #################################################################
    # n_frequency: int = field(
    #     default=1000,
    # )
    # scaling: float = field(
    #     default=150.0,
    # )
    # random_loc_seed: Optional[int] = field(
    #     default=777,
    # )
    # fan_in_fan_out: bool = field(
    #     default=False,
    # )
    # bias: str = field(
    #     default="none",
    # )
    # n_frequency_pattern: Optional[dict] = field(
    #     default_factory=dict,
    # )
    # init_weights: bool = field(
    #     default=False,
    # )
    #################################################################

    modules_to_save: Optional[list[str]] = field(
        default=None,
    )

    #################################################################
    
    target_modules: Optional[Union[list[str], str]] = field(
        default=None,
    )
    exclude_modules: Optional[Union[list[str], str]] = field(
        default=None,
    )
    layers_to_transform: Optional[Union[list[int], int]] = field(
        default=None,
    )
    layers_pattern: Optional[Union[list[str], str]] = field(
        default=None,
    )

    def __post_init__(self):
        super().__post_init__()
        self.peft_type = PeftType.GENFT
        self.target_modules = (
            set(self.target_modules) if isinstance(self.target_modules, list) else self.target_modules
        )
        self.exclude_modules = (
            set(self.exclude_modules) if isinstance(self.exclude_modules, list) else self.exclude_modules
        )
        if isinstance(self.target_modules, str) and self.layers_to_transform is not None:
            raise ValueError("`layers_to_transform` cannot be used when `target_modules` is a str.")
        if isinstance(self.target_modules, str) and self.layers_pattern is not None:
            raise ValueError("`layers_pattern` cannot be used when `target_modules` is a str.")
        if self.layers_pattern and not self.layers_to_transform:
            raise ValueError("When `layers_pattern` is specified, `layers_to_transform` must also be specified.")
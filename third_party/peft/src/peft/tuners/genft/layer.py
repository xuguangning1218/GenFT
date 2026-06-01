# Copyright 2024-present the HuggingFace Inc. team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import warnings
from typing import Any, Optional, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers.pytorch_utils import Conv1D

from peft.tuners.tuners_utils import BaseTunerLayer, check_adapters_to_merge

import math

class GenFTLayer(BaseTunerLayer):
    # All names of layers that may contain (trainable) adapter weights
    adapter_layer_names = ()
    # All names of other parameters that may contain adapter-related parameters
    other_param_names = ("genft_individual_features", "genft_scaling", "genft_individual_init")

    def __init__(self, base_layer: nn.Module, generator: nn.Module, **kwargs) -> None:
        self.base_layer = base_layer
        self.genft_individual_features = {}
        self.genft_individual_init_a = {}
        self.genft_individual_init_b = {}
        self.ratio_W0 = {}
        self.genft_scaling = {}
        self.individual_features = kwargs["individual_features"]
        self.drop = kwargs["drop"]
        if self.individual_features > 0:
            self.genft_individual_weight_a = nn.ParameterDict({})
            self.genft_individual_weight_b = nn.ParameterDict({})
        self.generator = generator
        
        # Mark the weight as unmerged
        self._disable_adapters = False
        self.merged_adapters = []
        self.kwargs = kwargs

        base_layer = self.get_base_layer()
        if isinstance(base_layer, nn.Linear):
            self.in_features, self.out_features = base_layer.in_features, base_layer.out_features
        elif isinstance(base_layer, Conv1D):
            self.in_features, self.out_features = (
                base_layer.weight.ds_shape if hasattr(base_layer.weight, "ds_shape") else base_layer.weight.shape
            )
        else:
            raise ValueError(f"Unsupported layer type {type(base_layer)}")

    def update_layer(self, adapter_name, scaling, individual_init_a, individual_init_b, ratio_W0):
        self.genft_scaling[adapter_name] = scaling
        self.genft_individual_features[adapter_name] = self.individual_features
        self.genft_individual_init_a[adapter_name] = individual_init_a
        self.genft_individual_init_b[adapter_name] = individual_init_b
        self.ratio_W0[adapter_name] = ratio_W0

        if self.individual_features > 0:
            in_features = self.base_layer.in_features
            out_features = self.base_layer.out_features
            self.genft_individual_weight_a[adapter_name] = nn.Parameter(torch.empty(self.individual_features, in_features))
            self.genft_individual_weight_b[adapter_name] = nn.Parameter(torch.empty(out_features, self.individual_features))
            self.reset_fourier_parameters(adapter_name)
            self._move_adapter_to_device_of_base_layer(adapter_name)
            self.set_adapter(self.active_adapters)


    @torch.no_grad()
    def reset_fourier_parameters(self, adapter_name):
        if adapter_name in self.genft_individual_weight_a.keys():
            init_method = self.genft_individual_init_a[adapter_name]
            if init_method == 'zero':
                nn.init.zeros_(self.genft_individual_weight_a[adapter_name])
            elif init_method == 'kaiming_uniform':
                nn.init.kaiming_uniform_(self.genft_individual_weight_a[adapter_name], a=math.sqrt(5))
            elif init_method == 'xavier_uniform':
                nn.init.xavier_uniform_(self.genft_individual_weight_a[adapter_name])
            elif init_method == 'normal':
                nn.init.normal_(self.genft_individual_weight_a[adapter_name], mean=0.0, std=0.01)
            else:
                raise ValueError(f"Unsupported initialization type for weight_a: {init_method}")
        
        if adapter_name in self.genft_individual_weight_b.keys():
            init_method = self.genft_individual_init_b[adapter_name]
            if init_method == 'zero':
                nn.init.zeros_(self.genft_individual_weight_b[adapter_name])
            elif init_method == 'kaiming_uniform':
                nn.init.kaiming_uniform_(self.genft_individual_weight_b[adapter_name], a=math.sqrt(5))
            elif init_method == 'xavier_uniform':
                nn.init.xavier_uniform_(self.genft_individual_weight_b[adapter_name])
            elif init_method == 'normal':
                nn.init.normal_(self.genft_individual_weight_b[adapter_name], mean=0.0, std=0.01)
            else:
                raise ValueError(f"Unsupported initialization type for weight_b: {init_method}")

    def get_delta_weight(self, adapter) -> torch.Tensor:
        base_device = self.base_layer.weight.device

        if self.genft_individual_features[adapter] > 0 and self.generator is not None:
            individual_weight_a = self.genft_individual_weight_a[adapter].to(base_device)
            individual_weight_b = self.genft_individual_weight_b[adapter].to(base_device)
            ratio_W0 = self.ratio_W0[adapter]
            delta_w = self.generator(ratio_W0 * self.base_layer.weight, individual_weight_a, individual_weight_b)
        elif self.genft_individual_features[adapter] == 0:
            delta_w = self.generator(self.base_layer.weight)
        elif self.generator is None:
            delta_w = torch.mm(self.genft_individual_weight_b[adapter], self.genft_individual_weight_a[adapter])
            delta_w = torch.dropout(delta_w, p=self.drop, train=True)
        return delta_w * self.genft_scaling[adapter]


class GenFTLinear(nn.Module, GenFTLayer):
    # GenFT implemented in a dense layer
    def __init__(
        self,
        base_layer,
        adapter_name: str,
        generator: nn.Module,
        **kwargs,
    ) -> None:
        super().__init__()
        GenFTLayer.__init__(self, base_layer, generator, **kwargs)
        self._active_adapter = adapter_name
        self.update_layer(adapter_name, kwargs["scaling"], kwargs["individual_init_a"], kwargs["individual_init_b"], kwargs["ratio_W0"])

    def merge(self, safe_merge: bool = False, adapter_names: Optional[list[str]] = None) -> None:
        """
        Merge the active adapter weights into the base weights

        Args:
            safe_merge (`bool`, *optional*):
                If True, the merge operation will be performed in a copy of the original weights and check for NaNs
                before merging the weights. This is useful if you want to check if the merge operation will produce
                NaNs. Defaults to `False`.
            adapter_names (`List[str]`, *optional*):
                The list of adapter names that should be merged. If None, all active adapters will be merged. Defaults
                to `None`.
        """
        print("##############################Merge#############################")
        adapter_names = check_adapters_to_merge(self, adapter_names)
        if not adapter_names:
            # no adapter to merge
            return

        for active_adapter in adapter_names:
            if active_adapter in self.fourierft_spectrum.keys():
                base_layer = self.get_base_layer()
                if safe_merge:
                    # Note that safe_merge will be slower than the normal merge
                    # because of the copy operation.
                    orig_weights = base_layer.weight.data.clone()
                    orig_weights += self.get_delta_weight(active_adapter)

                    if not torch.isfinite(orig_weights).all():
                        raise ValueError(
                            f"NaNs detected in the merged weights. The adapter {active_adapter} seems to be broken"
                        )

                    base_layer.weight.data = orig_weights
                else:
                    base_layer.weight.data += self.get_delta_weight(active_adapter)
                self.merged_adapters.append(active_adapter)

    def unmerge(self) -> None:
        """
        This method unmerges all merged adapter layers from the base weights.
        """
        print("##############################Unmerge#############################")
        if not self.merged:
            warnings.warn("Already unmerged. Nothing to do.")
            return
        while len(self.merged_adapters) > 0:
            active_adapter = self.merged_adapters.pop()
            if active_adapter in self.genft_individual_weight_a.keys():
                self.get_base_layer().weight.data -= self.get_delta_weight(active_adapter)

    def get_delta_weight(self, adapter) -> torch.Tensor:
        return super().get_delta_weight(adapter)

    def forward(self, x: torch.Tensor, *args: Any, **kwargs: Any) -> torch.Tensor:
        previous_dtype = x.dtype

        if self.disable_adapters:
            if self.merged:
                self.unmerge()
            result = self.base_layer(x, *args, **kwargs)
        elif self.merged:
            result = self.base_layer(x, *args, **kwargs)
        else:
            result = self.base_layer(x, *args, **kwargs)
            for active_adapter in self.active_adapters:

                base_device = self.base_layer.weight.device
                delta_w = self.get_delta_weight(active_adapter).to(base_device)
                x = x.to(delta_w.dtype)
                result = result + F.linear(x.to(base_device), delta_w)

        result = result.to(previous_dtype)
        return result

    def __repr__(self) -> str:
        rep = super().__repr__()
        return "genft." + rep

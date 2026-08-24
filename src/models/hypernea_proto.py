from typing import Dict

import torch
import torch.nn as nn


class HyperNEAPrototype(nn.Module):
    """HyperNEAT-inspired indirect encoding prototype for RF classification."""

    def __init__(
        self,
        classes: int,
        input_channels: int = 2,
        window_size: int = 256,
        hidden_size: int = 32,
        cppn_hidden: int = 24,
    ):
        super().__init__()

        self.classes = int(classes)
        self.input_channels = int(input_channels)
        self.window_size = int(window_size)
        self.hidden_size = int(hidden_size)
        self.cppn_hidden = int(cppn_hidden)

        self.register_buffer("input_coords", self._build_input_coords())
        self.register_buffer("hidden_coords", self._build_line_coords(self.hidden_size))
        self.register_buffer("output_coords", self._build_line_coords(self.classes))

        self.cppn_w1 = nn.Parameter(torch.empty(5, self.cppn_hidden), requires_grad=False)
        self.cppn_b1 = nn.Parameter(torch.empty(self.cppn_hidden), requires_grad=False)
        self.cppn_w2 = nn.Parameter(torch.empty(self.cppn_hidden, self.cppn_hidden), requires_grad=False)
        self.cppn_b2 = nn.Parameter(torch.empty(self.cppn_hidden), requires_grad=False)
        self.cppn_w3 = nn.Parameter(torch.empty(self.cppn_hidden, 1), requires_grad=False)
        self.cppn_b3 = nn.Parameter(torch.empty(1), requires_grad=False)
        self.weight_scale = nn.Parameter(torch.empty(1), requires_grad=False)
        self.bias_scale = nn.Parameter(torch.empty(1), requires_grad=False)

        self._decoded_cache = None
        self.reset_genome()

    def _build_input_coords(self) -> torch.Tensor:
        channel_axis = torch.linspace(-1.0, 1.0, steps=self.input_channels, dtype=torch.float32)
        time_axis = torch.linspace(-1.0, 1.0, steps=self.window_size, dtype=torch.float32)
        channel_grid, time_grid = torch.meshgrid(channel_axis, time_axis, indexing="ij")
        return torch.stack([time_grid.reshape(-1), channel_grid.reshape(-1)], dim=1)

    def _build_line_coords(self, size: int) -> torch.Tensor:
        axis = torch.linspace(-1.0, 1.0, steps=size, dtype=torch.float32)
        zeros = torch.zeros_like(axis)
        return torch.stack([axis, zeros], dim=1)

    def reset_genome(self, seed: int | None = None) -> None:
        generator = None
        if seed is not None:
            generator = torch.Generator(device="cpu")
            generator.manual_seed(seed)

        self.cppn_w1.data.normal_(mean=0.0, std=0.35, generator=generator)
        self.cppn_b1.data.normal_(mean=0.0, std=0.15, generator=generator)
        self.cppn_w2.data.normal_(mean=0.0, std=0.30, generator=generator)
        self.cppn_b2.data.normal_(mean=0.0, std=0.15, generator=generator)
        self.cppn_w3.data.normal_(mean=0.0, std=0.20, generator=generator)
        self.cppn_b3.data.zero_()
        self.weight_scale.data.fill_(0.35)
        self.bias_scale.data.fill_(0.15)
        self.invalidate_cache()

    def invalidate_cache(self) -> None:
        self._decoded_cache = None

    def get_genome(self) -> Dict[str, torch.Tensor]:
        return {
            "cppn_w1": self.cppn_w1.detach().cpu().clone(),
            "cppn_b1": self.cppn_b1.detach().cpu().clone(),
            "cppn_w2": self.cppn_w2.detach().cpu().clone(),
            "cppn_b2": self.cppn_b2.detach().cpu().clone(),
            "cppn_w3": self.cppn_w3.detach().cpu().clone(),
            "cppn_b3": self.cppn_b3.detach().cpu().clone(),
            "weight_scale": self.weight_scale.detach().cpu().clone(),
            "bias_scale": self.bias_scale.detach().cpu().clone(),
        }

    def clone_genome(self) -> Dict[str, torch.Tensor]:
        return self.get_genome()

    def set_genome(self, genome: Dict[str, torch.Tensor]) -> None:
        for name, value in genome.items():
            getattr(self, name).data.copy_(value.to(getattr(self, name).data.device))
        self.invalidate_cache()

    def _cppn(self, features: torch.Tensor) -> torch.Tensor:
        hidden = torch.tanh(features @ self.cppn_w1 + self.cppn_b1)
        hidden = torch.tanh(hidden @ self.cppn_w2 + self.cppn_b2)
        output = torch.tanh(hidden @ self.cppn_w3 + self.cppn_b3)
        return output.squeeze(-1)

    def _generate_weight_matrix(self, source_coords: torch.Tensor, target_coords: torch.Tensor) -> torch.Tensor:
        source = source_coords.unsqueeze(0).expand(target_coords.shape[0], -1, -1)
        target = target_coords.unsqueeze(1).expand(-1, source_coords.shape[0], -1)
        delta = target - source
        radius = torch.sqrt(torch.clamp(delta[..., 0].square() + delta[..., 1].square(), min=1e-8))
        features = torch.stack(
            [source[..., 0], source[..., 1], target[..., 0], target[..., 1], radius],
            dim=-1,
        )
        weights = self._cppn(features.reshape(-1, 5)).reshape(target_coords.shape[0], source_coords.shape[0])
        return weights * self.weight_scale

    def _generate_bias(self, target_coords: torch.Tensor) -> torch.Tensor:
        zeros = torch.zeros_like(target_coords)
        delta = target_coords - zeros
        radius = torch.sqrt(torch.clamp(delta[..., 0].square() + delta[..., 1].square(), min=1e-8))
        features = torch.stack(
            [zeros[..., 0], zeros[..., 1], target_coords[..., 0], target_coords[..., 1], radius],
            dim=-1,
        )
        bias = self._cppn(features)
        return bias * self.bias_scale

    def _decode(self, device: torch.device) -> Dict[str, torch.Tensor]:
        input_coords = self.input_coords.to(device)
        hidden_coords = self.hidden_coords.to(device)
        output_coords = self.output_coords.to(device)
        return {
            "w1": self._generate_weight_matrix(input_coords, hidden_coords),
            "b1": self._generate_bias(hidden_coords),
            "w2": self._generate_weight_matrix(hidden_coords, output_coords),
            "b2": self._generate_bias(output_coords),
        }

    def decoded_parameters(self, device: torch.device) -> Dict[str, torch.Tensor]:
        if self._decoded_cache is None or self._decoded_cache["w1"].device != device:
            self._decoded_cache = self._decode(device)
        return self._decoded_cache

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        params = self.decoded_parameters(x.device)
        flat = x.reshape(x.shape[0], -1)
        hidden = torch.tanh(flat @ params["w1"].t() + params["b1"])
        return hidden @ params["w2"].t() + params["b2"]
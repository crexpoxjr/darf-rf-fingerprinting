import torch
import torch.nn as nn


class RF_CNN_MULTI(nn.Module):
	"""Multi-channel 1D CNN expecting [B, C, T] with C=4 (I, Q, |IQ|, phase)."""

	def __init__(
		self,
		classes: int = 5,
		in_channels: int = 4,
	):
		super().__init__()

		if in_channels != 4:
			raise ValueError(
				f"RF_CNN_MULTI expects 4 input channels (I, Q, magnitude, phase), got {in_channels}"
			)

		self.features = nn.Sequential(
			nn.Conv1d(in_channels, 32, kernel_size=7, padding=3),
			nn.BatchNorm1d(32),
			nn.ReLU(),
			nn.MaxPool1d(2),

			nn.Conv1d(32, 64, kernel_size=5, padding=2),
			nn.BatchNorm1d(64),
			nn.ReLU(),
			nn.MaxPool1d(2),

			nn.Conv1d(64, 128, kernel_size=3, padding=1),
			nn.BatchNorm1d(128),
			nn.ReLU(),
			nn.AdaptiveMaxPool1d(4),
		)

		self.classifier = nn.Sequential(
			nn.Flatten(),
			nn.Dropout(0.25),
			nn.Linear(128 * 4, 128),
			nn.ReLU(),
			nn.Dropout(0.25),
			nn.Linear(128, classes),
		)

	def forward(self, x: torch.Tensor) -> torch.Tensor:
		x = self.features(x)
		return self.classifier(x)

"""CNN baseline over HOG block grids (and hybrid geometric+HOG fusion)."""

from __future__ import annotations

try:
    import torch
    import torch.nn as nn
except ImportError:  # pragma: no cover
    torch = None  # type: ignore[assignment]
    nn = None  # type: ignore[assignment]


def infer_cnn_shape(input_dim: int, channels: int = 1) -> tuple[int, int, int]:
    """Legacy fallback: factor *input_dim* into (C, H, W) — not used for HOG/hybrid."""
    side = int(round(input_dim**0.5))
    while side > 1 and input_dim % side != 0:
        side -= 1
    if side <= 1:
        return (1, 1, input_dim)
    other = input_dim // side
    return (channels, side, other)


def _hog_tensor_channels_first(
    hog_flat: torch.Tensor,
    grid: tuple[int, int, int],
) -> torch.Tensor:
    """(batch, hog_dim) -> (batch, features_per_block, block_rows, block_cols)."""
    br, bc, bpf = grid
    batch = hog_flat.size(0)
    blocks = hog_flat.view(batch, br, bc, bpf)
    return blocks.permute(0, 3, 1, 2).contiguous()


class HogCNNClassifier(nn.Module):
    """Convolutions on the HOG block grid (channels = features per block)."""

    def __init__(
        self,
        hog_grid: tuple[int, int, int],
        n_classes: int,
        *,
        hidden_channels: int = 32,
    ) -> None:
        super().__init__()
        bpf, br, bc = hog_grid[2], hog_grid[0], hog_grid[1]
        self.hog_grid = hog_grid
        self.reshape = (bpf, br, bc)
        self.conv = nn.Sequential(
            nn.Conv2d(bpf, hidden_channels, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(hidden_channels, hidden_channels * 2, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        self.fc = nn.Linear(hidden_channels * 2, n_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 2:
            x = _hog_tensor_channels_first(x, self.hog_grid)
        x = self.conv(x)
        return self.fc(x.view(x.size(0), -1))


class HybridCNNClassifier(nn.Module):
    """CNN trunk on HOG blocks + MLP on geometric descriptors (flat hybrid vector in)."""

    def __init__(
        self,
        geom_dim: int,
        hog_grid: tuple[int, int, int],
        n_classes: int,
        *,
        hidden_channels: int = 32,
        geom_hidden: int = 128,
    ) -> None:
        super().__init__()
        self.geom_dim = geom_dim
        self.hog_grid = hog_grid
        br, bc, bpf = hog_grid
        self.hog_cnn = nn.Sequential(
            nn.Conv2d(bpf, hidden_channels, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(hidden_channels, hidden_channels * 2, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        self.geom_branch = nn.Sequential(
            nn.Linear(geom_dim, geom_hidden),
            nn.ReLU(),
            nn.Dropout(0.2),
        )
        self.head = nn.Linear(hidden_channels * 2 + geom_hidden, n_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        geom = x[:, : self.geom_dim]
        hog = _hog_tensor_channels_first(x[:, self.geom_dim :], self.hog_grid)
        hog_vec = self.hog_cnn(hog).view(x.size(0), -1)
        geom_vec = self.geom_branch(geom)
        return self.head(torch.cat([hog_vec, geom_vec], dim=1))


class CNNClassifier(nn.Module):
    """Legacy CNN on arbitrary (C, H, W) — kept for unknown flat layouts."""

    def __init__(
        self,
        shape: tuple[int, int, int],
        n_classes: int,
        *,
        hidden_channels: int = 32,
    ) -> None:
        super().__init__()
        c, h, w = shape
        self.shape = shape
        self.conv = nn.Sequential(
            nn.Conv2d(c, hidden_channels, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(hidden_channels, hidden_channels * 2, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        self.fc = nn.Linear(hidden_channels * 2, n_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 2:
            x = x.view(x.size(0), *self.shape)
        x = self.conv(x)
        return self.fc(x.view(x.size(0), -1))


def build_cnn(
    input_dim: int,
    n_classes: int,
    *,
    channels: int = 1,
    hidden_channels: int = 32,
    geom_dim: int = 0,
    hog_grid: tuple[int, int, int] | None = None,
    layout: str | None = None,
) -> tuple[nn.Module, tuple[int, ...] | None]:
    if torch is None:
        raise ImportError("PyTorch is required. Install with: pip install torch")

    kind = layout or ("hybrid" if geom_dim > 0 and hog_grid else ("hog" if hog_grid else "flat"))

    if kind == "hybrid" and hog_grid is not None and geom_dim > 0:
        model = HybridCNNClassifier(geom_dim, hog_grid, n_classes, hidden_channels=hidden_channels)
        return model, None

    if kind == "hog" and hog_grid is not None:
        model = HogCNNClassifier(hog_grid, n_classes, hidden_channels=hidden_channels)
        return model, model.reshape

    shape = infer_cnn_shape(input_dim, channels=channels)
    return CNNClassifier(shape, n_classes, hidden_channels=hidden_channels), shape

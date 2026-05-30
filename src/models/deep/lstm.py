"""LSTM baseline over HOG block sequences (and hybrid fusion)."""

from __future__ import annotations

try:
    import torch
    import torch.nn as nn
except ImportError:  # pragma: no cover
    torch = None  # type: ignore[assignment]
    nn = None  # type: ignore[assignment]


def infer_lstm_shape(input_dim: int, seq_len: int | None = None) -> tuple[int, int]:
    """Legacy fallback factorization of a flat vector."""
    if seq_len is not None and input_dim % seq_len == 0:
        return (seq_len, input_dim // seq_len)
    seq_len = max(2, int(round(input_dim**0.5)))
    while seq_len > 1 and input_dim % seq_len != 0:
        seq_len -= 1
    return (seq_len, input_dim // seq_len)


def _hog_sequence(hog_flat: torch.Tensor, grid: tuple[int, int, int]) -> torch.Tensor:
    """(batch, hog_dim) -> (batch, seq_len, feat_dim) with seq_len = block_rows * block_cols."""
    br, bc, bpf = grid
    batch = hog_flat.size(0)
    return hog_flat.view(batch, br * bc, bpf)


class HogLSTMClassifier(nn.Module):
    def __init__(
        self,
        hog_grid: tuple[int, int, int],
        n_classes: int,
        *,
        hidden_size: int = 128,
        num_layers: int = 1,
    ) -> None:
        super().__init__()
        br, bc, bpf = hog_grid
        self.hog_grid = hog_grid
        self.reshape = (br * bc, bpf)
        self.lstm = nn.LSTM(
            input_size=bpf,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
        )
        self.fc = nn.Linear(hidden_size, n_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 2:
            x = _hog_sequence(x, self.hog_grid)
        out, _ = self.lstm(x)
        return self.fc(out[:, -1, :])


class HybridLSTMClassifier(nn.Module):
    """LSTM over HOG block sequence + geometric MLP branch."""

    def __init__(
        self,
        geom_dim: int,
        hog_grid: tuple[int, int, int],
        n_classes: int,
        *,
        hidden_size: int = 128,
        num_layers: int = 1,
        geom_hidden: int = 128,
    ) -> None:
        super().__init__()
        self.geom_dim = geom_dim
        self.hog_grid = hog_grid
        _, _, bpf = hog_grid
        self.lstm = nn.LSTM(
            input_size=bpf,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
        )
        self.geom_branch = nn.Sequential(
            nn.Linear(geom_dim, geom_hidden),
            nn.ReLU(),
            nn.Dropout(0.2),
        )
        self.head = nn.Linear(hidden_size + geom_hidden, n_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        geom = x[:, : self.geom_dim]
        hog = _hog_sequence(x[:, self.geom_dim :], self.hog_grid)
        out, _ = self.lstm(hog)
        return self.head(torch.cat([out[:, -1, :], self.geom_branch(geom)], dim=1))


class LSTMClassifier(nn.Module):
    """Legacy LSTM on arbitrary (seq_len, feat_dim) reshape."""

    def __init__(
        self,
        seq_len: int,
        feat_dim: int,
        n_classes: int,
        *,
        hidden_size: int = 128,
        num_layers: int = 1,
    ) -> None:
        super().__init__()
        self.seq_len = seq_len
        self.feat_dim = feat_dim
        self.lstm = nn.LSTM(
            input_size=feat_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
        )
        self.fc = nn.Linear(hidden_size, n_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 2:
            x = x.view(x.size(0), self.seq_len, self.feat_dim)
        out, _ = self.lstm(x)
        return self.fc(out[:, -1, :])


def build_lstm(
    input_dim: int,
    n_classes: int,
    *,
    seq_len: int | None = None,
    hidden_size: int = 128,
    num_layers: int = 1,
    geom_dim: int = 0,
    hog_grid: tuple[int, int, int] | None = None,
    layout: str | None = None,
) -> tuple[nn.Module, tuple[int, ...] | None]:
    if torch is None:
        raise ImportError("PyTorch is required. Install with: pip install torch")

    kind = layout or ("hybrid" if geom_dim > 0 and hog_grid else ("hog" if hog_grid else "flat"))

    if kind == "hybrid" and hog_grid is not None and geom_dim > 0:
        return (
            HybridLSTMClassifier(
                geom_dim,
                hog_grid,
                n_classes,
                hidden_size=hidden_size,
                num_layers=num_layers,
            ),
            None,
        )

    if kind == "hog" and hog_grid is not None:
        model = HogLSTMClassifier(hog_grid, n_classes, hidden_size=hidden_size, num_layers=num_layers)
        return model, model.reshape

    shape = infer_lstm_shape(input_dim, seq_len=seq_len)
    seq_len_v, feat_dim = shape
    return (
        LSTMClassifier(seq_len_v, feat_dim, n_classes, hidden_size=hidden_size, num_layers=num_layers),
        shape,
    )

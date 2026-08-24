"""FT-Transformer (Feature-Tokenizer Transformer) backbone for tabular band-gap regression.

v2 improvements for tabular/regression competitiveness:
- Optional PeriodicTokenizer (sin/cos Fourier features) for continuous inputs.
- Pre-LayerNorm transformer blocks with optional LayerScale residual gating.
- Optional ReGLU feed-forward network.
- Optional Wide & Deep head: concatenates raw input features with the pooled
  attention token, so the model retains linear signal while attention adds on top.
"""
import torch
import torch.nn as nn


class FeatureTokenizer(nn.Module):
    """Maps each continuous feature to a d-dimensional token embedding (linear)."""

    def __init__(self, n_features: int, d_model: int, add_cls: bool = True):
        super().__init__()
        self.n_features = n_features
        self.add_cls = add_cls
        self.linear = nn.Linear(1, d_model)
        if add_cls:
            self.cls_token = nn.Parameter(torch.randn(1, 1, d_model) * 0.02)

    def forward(self, x):
        tokens = torch.stack(
            [self.linear(x[:, i : i + 1]) for i in range(self.n_features)], dim=1
        )
        if self.add_cls:
            cls = self.cls_token.expand(x.size(0), -1, -1)
            tokens = torch.cat([cls, tokens], dim=1)
        return tokens


class PeriodicTokenizer(nn.Module):
    """Fourier (sin/cos) feature tokenizer. Significantly boosts R^2 on
    continuous tabular data by giving the transformer smooth non-linear input
    representations instead of straight-line projections."""

    def __init__(self, n_features: int, d_model: int, n_frequencies: int = 16,
                 add_cls: bool = True):
        super().__init__()
        self.n_features = n_features
        self.n_frequencies = n_frequencies
        self.add_cls = add_cls
        # Learnable per-feature frequencies and phases.
        self.frequencies = nn.Parameter(torch.randn(n_features, n_frequencies) * 0.1)
        self.phases = nn.Parameter(torch.zeros(n_features, n_frequencies))
        self.proj = nn.ModuleList(
            [nn.Linear(2 * n_frequencies, d_model) for _ in range(n_features)]
        )
        if add_cls:
            self.cls_token = nn.Parameter(torch.randn(1, 1, d_model) * 0.02)

    def forward(self, x):
        # x: (B, N)
        x_exp = x.unsqueeze(-1)  # (B, N, 1)
        args = 2 * torch.pi * self.frequencies * x_exp + self.phases  # (B, N, F)
        periodic = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)  # (B, N, 2F)
        tokens = torch.stack(
            [self.proj[i](periodic[:, i, :]) for i in range(self.n_features)], dim=1
        )
        if self.add_cls:
            cls = self.cls_token.expand(x.size(0), -1, -1)
            tokens = torch.cat([cls, tokens], dim=1)
        return tokens


class ReGLUFFN(nn.Module):
    """Gated feed-forward (ReGLU): ReLU(W1 x) * (W2 x)."""

    def __init__(self, d_model: int, d_ffn: int, dropout: float = 0.1):
        super().__init__()
        self.linear1 = nn.Linear(d_model, d_ffn * 2)
        self.linear2 = nn.Linear(d_ffn, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x1, x2 = self.linear1(x).chunk(2, dim=-1)
        x = torch.relu(x1) * x2
        return self.linear2(self.dropout(x))


class TransformerEncoderBlock(nn.Module):
    """Pre-LayerNorm self-attention block with optional LayerScale + ReGLU FFN.

    Pre-LayerNorm (normalize before the sublayer) keeps an unimpeded gradient
    highway along the residual path, which is critical for stable training on
    small tabular datasets. LayerScale gates each residual with a learnable
    gamma initialized small, letting deeper nets warm up gradually.
    """

    def __init__(self, d_model: int, n_heads: int, d_ffn: int, dropout: float = 0.1,
                 use_reglu: bool = True, layer_scale: float = 1e-2):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.norm2 = nn.LayerNorm(d_model)
        if use_reglu:
            self.ffn = ReGLUFFN(d_model, d_ffn, dropout)
        else:
            self.ffn = nn.Sequential(
                nn.Linear(d_model, d_ffn), nn.ReLU(), nn.Dropout(dropout),
                nn.Linear(d_ffn, d_model),
            )
        self.dropout = nn.Dropout(dropout)
        self.layer_scale = layer_scale
        if layer_scale > 0:
            self.gamma_1 = nn.Parameter(layer_scale * torch.ones(d_model))
            self.gamma_2 = nn.Parameter(layer_scale * torch.ones(d_model))

    def forward(self, x, need_weights=False):
        a = self.norm1(x)
        attn_out, weights = self.attn(a, a, a, need_weights=need_weights,
                                      average_attn_weights=True)
        if self.layer_scale > 0:
            x = x + self.gamma_1 * self.dropout(attn_out)
        else:
            x = x + self.dropout(attn_out)
        f = self.norm2(x)
        if self.layer_scale > 0:
            x = x + self.gamma_2 * self.dropout(self.ffn(f))
        else:
            x = x + self.dropout(self.ffn(f))
        return x, weights


class PG_SANN(nn.Module):
    """Physics-Guided Self-Attention Neural Network (FT-Transformer, v2).

    Args:
        n_features: number of input continuous features.
        d_model: token embedding / attention dimension.
        n_heads, n_layers, d_ffn, dropout: transformer config.
        pool: 'cls' (CLS token) or 'mean'.
        tokenizer: 'linear' or 'periodic'.
        use_reglu: use gated FFN.
        layer_scale: LayerScale init value (>0 enables).
        deep_head: if True, concat raw input features with pooled token (Wide & Deep).
        head_width: hidden width of the MLP regression head.
    """

    def __init__(self, n_features: int, d_model: int = 64, n_heads: int = 4,
                 n_layers: int = 3, d_ffn: int = 128, dropout: float = 0.1,
                 pool: str = "mean", tokenizer: str = "periodic",
                 use_reglu: bool = True, layer_scale: float = 1e-2,
                 deep_head: bool = True, head_width: int = 256):
        super().__init__()
        self.n_features = n_features
        self.d_model = d_model
        self.n_heads = n_heads
        self.n_layers = n_layers
        self.pool = pool
        self.deep_head = deep_head
        self.head_width = head_width

        tok_cls = PeriodicTokenizer if tokenizer == "periodic" else FeatureTokenizer
        self.tokenizer = tok_cls(n_features, d_model, add_cls=(pool == "cls"))

        self.blocks = nn.ModuleList(
            [TransformerEncoderBlock(d_model, n_heads, d_ffn, dropout,
                                     use_reglu=use_reglu, layer_scale=layer_scale)
             for _ in range(n_layers)]
        )
        self.pool_norm = nn.LayerNorm(d_model)

        # Wide & Deep head: concat pooled attention (d_model) with raw features (n_features).
        in_dim = d_model + (n_features if deep_head else 0)
        self.head = nn.Sequential(
            nn.LayerNorm(in_dim),
            nn.Linear(in_dim, head_width),
            nn.ReLU(),
            nn.Linear(head_width, head_width),
            nn.ReLU(),
            nn.Linear(head_width, 1),
        )

    def forward(self, x, return_attention=False):
        tokens = self.tokenizer(x)
        attention_maps = []
        h = tokens
        for block in self.blocks:
            h, w = block(h, need_weights=return_attention)
            if return_attention:
                attention_maps.append(w)
        if self.pool == "cls":
            pooled = h[:, 0]
        else:
            pooled = h.mean(dim=1)
        pooled = self.pool_norm(pooled)
        if self.deep_head:
            z = torch.cat([pooled, x], dim=1)
        else:
            z = pooled
        out = self.head(z).squeeze(-1)
        if return_attention:
            return out, attention_maps
        return out


def build_model(n_features: int, cfg) -> PG_SANN:
    m = cfg.get("model", {})
    return PG_SANN(
        n_features=n_features,
        d_model=int(m.get("d_model", 64)),
        n_heads=int(m.get("n_heads", 4)),
        n_layers=int(m.get("n_layers", 3)),
        d_ffn=int(m.get("d_ffn", 128)),
        dropout=float(m.get("dropout", 0.1)),
        pool=str(m.get("pool", "mean")),
        tokenizer=str(m.get("tokenizer", "periodic")),
        use_reglu=bool(m.get("use_reglu", True)),
        layer_scale=float(m.get("layer_scale", 1e-2)),
        deep_head=bool(m.get("deep_head", True)),
        head_width=int(m.get("head_width", 256)),
    )

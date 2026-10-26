"""
Model components for conditional order-book generation (pure PyTorch, no NumPy).

Current contents:
    ConditionalGRU      GRU stack whose input at every step is the step's
                        features concatenated with a ticker embedding; shared
                        building block for the baseline and (later) TimeGAN.
    RCGANGenerator      noise + ticker -> feature sequence in [0, 1].
    RCGANDiscriminator  feature sequence + ticker -> one real/fake logit per step.
    sample_noise        noise sequences for the generators.

Baseline: RCGAN-style recurrent conditional GAN, adapted from
    Esteban, Hyland & Raetsch (2017), "Real-valued (Medical) Time Series
    Generation with Recurrent Conditional GANs", arXiv:1706.02633.
Conditioning by concatenating a learned label embedding follows
    Mirza & Osindero (2014), "Conditional Generative Adversarial Nets",
    arXiv:1411.1784.
For a fair comparison with TimeGAN (Yoon, Jarrett & van der Schaar, NeurIPS
2019), the baseline uses TimeGAN's 3-layer GRU backbone, as the TimeGAN authors
did for their benchmarks, and a bidirectional discriminator (TimeGAN paper,
Section 4.2). It is therefore an RCGAN-*style* adaptation, not an exact
reproduction of the original (LSTM) implementation.

Shapes: B = batch, T = sequence length, F = features per step,
Z = noise size, C = ticker-embedding size, H = hidden size.
"""

import torch
from torch import nn


def sample_noise(
    batch_size, seq_len, z_dim, distribution="uniform", device=None
):
    """
    Draw independent noise for every sequence, step and dimension.

    Args:
        batch_size (int): B.
        seq_len (int): T; may differ from the training window length, since the
            GRUs are length-agnostic (e.g. 500 steps at generation time).
        z_dim (int): Z.
        distribution (str): "uniform" for U(0, 1), as in the TimeGAN reference
            code, or "normal" for N(0, 1).
        device: torch device for the result.

    Returns:
        torch.Tensor (B, T, Z).

    Raises:
        ValueError: for an unknown distribution.
    """
    shape = (batch_size, seq_len, z_dim)
    if distribution == "uniform":
        return torch.rand(shape, device=device)
    if distribution == "normal":
        return torch.randn(shape, device=device)
    raise ValueError(f"unknown noise distribution {distribution!r}")


class ConditionalGRU(nn.Module):
    """
    GRU stack conditioned on the ticker at every time step, plus a linear head.

    For each step t the GRU input is [x_t, c], where c = Embedding(ticker_id)
    is repeated over time (Mirza & Osindero 2014: the condition is given to the
    network as an extra input). A linear layer maps each hidden state to the
    output size, followed by an optional activation.

    Args:
        input_dim (int): size of x_t.
        output_dim (int): size of each output step.
        n_tickers (int): number of distinct ticker ids.
        cond_dim (int): C, ticker-embedding size.
        hidden_dim (int): H, GRU hidden size.
        num_layers (int): stacked GRU layers (3 in the TimeGAN paper).
        bidirectional (bool): run the GRU in both directions (discriminators).
        activation (nn.Module | None): applied to the head output, e.g. Sigmoid.
    """

    def __init__(
        self,
        input_dim,
        output_dim,
        n_tickers,
        cond_dim=8,
        hidden_dim=24,
        num_layers=3,
        bidirectional=False,
        activation=None,
    ):
        super().__init__()
        self.embedding = nn.Embedding(n_tickers, cond_dim)
        self.gru = nn.GRU(
            input_size=input_dim + cond_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=bidirectional,
        )
        self.head = nn.Linear(
            hidden_dim * (2 if bidirectional else 1), output_dim
        )
        self.activation = (
            activation if activation is not None else nn.Identity()
        )

    def forward(self, x, ticker_ids):
        """
        Args:
            x (torch.Tensor): (B, T, input_dim) input sequence.
            ticker_ids (torch.LongTensor): (B,) ticker ids.

        Returns:
            torch.Tensor (B, T, output_dim).
        """
        c = self.embedding(ticker_ids)  # (B, C)
        c = c.unsqueeze(1).expand(
            -1, x.size(1), -1
        )  # (B, T, C), same c every step
        h, _ = self.gru(torch.cat([x, c], dim=-1))  # (B, T, H or 2H)
        return self.activation(self.head(h))


class RCGANGenerator(nn.Module):
    """
    RCGAN generator: maps noise z_t and the ticker to features, step by step.

    Unlike C-RNN-GAN, it never sees its own previous output: temporal structure
    can only come from the GRU's hidden state. It outputs directly in the
    (min-max scaled) feature space, so the sigmoid matches the [0, 1] data.
    There is no embedder, recovery, supervisor or supervised loss: these are
    exactly the TimeGAN components the comparison tests.

    Args:
        z_dim (int): Z, noise size per step.
        feature_dim (int): F, output features (41 unconstrained, 40 constrained).
        n_tickers (int): number of ticker ids.
        cond_dim, hidden_dim, num_layers: see ConditionalGRU.
    """

    def __init__(
        self,
        z_dim,
        feature_dim,
        n_tickers,
        cond_dim=8,
        hidden_dim=24,
        num_layers=3,
    ):
        super().__init__()
        self.z_dim = z_dim
        self.net = ConditionalGRU(
            z_dim,
            feature_dim,
            n_tickers,
            cond_dim,
            hidden_dim,
            num_layers,
            activation=nn.Sigmoid(),
        )

    def forward(self, z, ticker_ids):
        """
        Args:
            z (torch.Tensor): (B, T, Z) noise.
            ticker_ids (torch.LongTensor): (B,) ticker ids to generate.

        Returns:
            torch.Tensor (B, T, F) synthetic scaled features in (0, 1).
        """
        return self.net(z, ticker_ids)


class RCGANDiscriminator(nn.Module):
    """
    RCGAN discriminator: one real/fake logit per step of a feature sequence.

    The ticker embedding is part of the input, so the discriminator can reject
    sequences that look real but belong to a different stock; without it the
    generator could ignore the condition. Bidirectional, matching TimeGAN's
    discriminator so only the components under test differ. Outputs raw logits
    (no sigmoid), to be used with BCEWithLogitsLoss for numerical stability.

    Args:
        feature_dim (int): F.
        n_tickers (int): number of ticker ids.
        cond_dim, hidden_dim, num_layers: see ConditionalGRU.
    """

    def __init__(
        self, feature_dim, n_tickers, cond_dim=8, hidden_dim=24, num_layers=3
    ):
        super().__init__()
        self.net = ConditionalGRU(
            feature_dim,
            1,
            n_tickers,
            cond_dim,
            hidden_dim,
            num_layers,
            bidirectional=True,
        )

    def forward(self, x, ticker_ids):
        """
        Args:
            x (torch.Tensor): (B, T, F) real or synthetic scaled features.
            ticker_ids (torch.LongTensor): (B,) ticker ids.

        Returns:
            torch.Tensor (B, T) logits; > 0 means "looks real".
        """
        return self.net(x, ticker_ids).squeeze(-1)


def count_parameters(module):
    """Number of trainable parameters in a module (for the resource table)."""
    return sum(p.numel() for p in module.parameters() if p.requires_grad)

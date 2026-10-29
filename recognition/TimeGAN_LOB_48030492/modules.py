"""
Model components for conditional order-book generation (pure PyTorch, no NumPy).

Contents, in file order:
    Shared building blocks
        sample_noise        noise sequences for the generators.
        ConditionalGRU      GRU stack whose input at every step is the step's
                            features concatenated with a ticker embedding.
    RCGAN-style baseline (feature space)
        RCGANGenerator      noise + ticker -> feature sequence in [0, 1].
        RCGANDiscriminator  feature sequence + ticker -> real/fake logit per step.
    TimeGAN (latent space)
        Embedder            feature sequence + ticker -> latent sequence.
        Recovery            latent sequence + ticker -> feature sequence.
        Generator           noise + ticker -> latent sequence.
        Supervisor          latent sequence + ticker -> next-step latents.
        Discriminator       latent sequence + ticker -> real/fake logit per step.
    Utilities
        count_parameters    trainable parameter count.

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

TimeGAN networks follow Yoon, Jarrett & van der Schaar (2019), "Time-series
Generative Adversarial Networks", NeurIPS, and the authors' reference code
(github.com/jsyoon0823/TimeGAN, timegan.py). The reference code is
unconditional; the ticker conditioning on every network is this project's
extension.

Shapes: B = batch, T = sequence length, F = features per step,
Z = noise size, C = ticker-embedding size, H = hidden size, K = latent size.
"""

import torch
from torch import nn


# =============================================================================
# Shared building blocks (used by both RCGAN and TimeGAN)
# =============================================================================


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

    cond_dim=0 switches conditioning off: no embedding is created and the GRU
    sees x_t alone, so the ticker id is ignored (unconditional ablation and
    the single-ticker fallback). A zero-width nn.Embedding is avoided on
    purpose: it works on CPU but crashes CUDA with an illegal memory access.

    Args:
        input_dim (int): size of x_t.
        output_dim (int): size of each output step.
        n_tickers (int): number of distinct ticker ids.
        cond_dim (int): C, ticker-embedding size; 0 disables conditioning.
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
        # None when unconditional (cond_dim=0); see the class docstring.
        self.embedding = nn.Embedding(n_tickers, cond_dim) if cond_dim > 0 else None
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
            ticker_ids (torch.LongTensor): (B,) ticker ids (ignored when
                unconditional).

        Returns:
            torch.Tensor (B, T, output_dim).
        """
        if self.embedding is not None:
            c = self.embedding(ticker_ids)  # (B, C)
            c = c.unsqueeze(1).expand(
                -1, x.size(1), -1
            )  # (B, T, C), same c every step
            x = torch.cat([x, c], dim=-1)  # (B, T, input_dim + C)
        h, _ = self.gru(x)  # (B, T, H or 2H)
        return self.activation(self.head(h))


# =============================================================================
# RCGAN-style baseline (Esteban et al. 2017): works directly in feature space
# =============================================================================


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


# =============================================================================
# TimeGAN (Yoon et al. 2019): adversarial game in a learned latent space
# =============================================================================


class Embedder(nn.Module):
    """
    TimeGAN embedding network E: maps real feature sequences to latent sequences.

    The adversarial game and the supervisor both work in this latent space
    rather than on raw features (TimeGAN paper, Section 4.1). As in the
    reference code, the latent output passes through a sigmoid, so latents lie
    in (0, 1) like the generator's output and the two can be compared directly.
    The GRU runs forwards only, so latent step t depends on features up to t.

    Args:
        feature_dim (int): F, input features (41 unconstrained, 40 constrained).
        n_tickers (int): number of ticker ids.
        latent_dim (int | None): K; None uses hidden_dim, as the reference
            code does.
        cond_dim, hidden_dim, num_layers: see ConditionalGRU.
    """

    def __init__(
        self,
        feature_dim,
        n_tickers,
        latent_dim=None,
        cond_dim=8,
        hidden_dim=24,
        num_layers=3,
    ):
        super().__init__()
        self.latent_dim = latent_dim or hidden_dim
        self.net = ConditionalGRU(
            feature_dim,
            self.latent_dim,
            n_tickers,
            cond_dim,
            hidden_dim,
            num_layers,
            activation=nn.Sigmoid(),
        )

    def forward(self, x, ticker_ids):
        """
        Args:
            x (torch.Tensor): (B, T, F) scaled features.
            ticker_ids (torch.LongTensor): (B,) ticker ids.

        Returns:
            torch.Tensor (B, T, K) latent sequence in (0, 1).
        """
        return self.net(x, ticker_ids)


class Recovery(nn.Module):
    """
    TimeGAN recovery network R: maps latent sequences back to feature sequences.

    Trained with E as an autoencoder (phase 1, reconstruction loss
    ||x - R(E(x))||^2), then decodes the generator's latents into features at
    generation time. The sigmoid matches the [0, 1] min-max scaled features.

    Args:
        feature_dim (int): F, output features.
        n_tickers (int): number of ticker ids.
        latent_dim (int | None): K; must match the Embedder's latent_dim.
        cond_dim, hidden_dim, num_layers: see ConditionalGRU.
    """

    def __init__(
        self,
        feature_dim,
        n_tickers,
        latent_dim=None,
        cond_dim=8,
        hidden_dim=24,
        num_layers=3,
    ):
        super().__init__()
        self.latent_dim = latent_dim or hidden_dim
        self.net = ConditionalGRU(
            self.latent_dim,
            feature_dim,
            n_tickers,
            cond_dim,
            hidden_dim,
            num_layers,
            activation=nn.Sigmoid(),
        )

    def forward(self, h, ticker_ids):
        """
        Args:
            h (torch.Tensor): (B, T, K) latent sequence.
            ticker_ids (torch.LongTensor): (B,) ticker ids.

        Returns:
            torch.Tensor (B, T, F) reconstructed scaled features in (0, 1).
        """
        return self.net(h, ticker_ids)


class Generator(nn.Module):
    """
    TimeGAN generator G: maps noise and the ticker to latent sequences.

    Same structure as RCGANGenerator, but it outputs in the latent space
    learned by the Embedder rather than in feature space; Recovery turns its
    output into features. The sigmoid matches the Embedder's (0, 1) latents.
    During joint training its output is passed through the Supervisor, so G
    only has to get the overall shape right while S supplies the step-to-step
    dynamics (TimeGAN paper, Section 4.2).

    Args:
        z_dim (int): Z, noise size per step.
        n_tickers (int): number of ticker ids.
        latent_dim (int | None): K; must match the Embedder's latent_dim.
        cond_dim, hidden_dim, num_layers: see ConditionalGRU.
    """

    def __init__(
        self,
        z_dim,
        n_tickers,
        latent_dim=None,
        cond_dim=8,
        hidden_dim=24,
        num_layers=3,
    ):
        super().__init__()
        self.z_dim = z_dim
        self.latent_dim = latent_dim or hidden_dim
        self.net = ConditionalGRU(
            z_dim,
            self.latent_dim,
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
            torch.Tensor (B, T, K) synthetic latent sequence in (0, 1).
        """
        return self.net(z, ticker_ids)


class Supervisor(nn.Module):
    """
    TimeGAN supervisor S: predicts the next latent step from the latents so far.

    Output step t is the prediction of latent step t + 1, given steps 1..t (the
    GRU runs forwards only, so it cannot look ahead). The supervised loss is
    therefore ||h[:, 1:] - S(h)[:, :-1]||^2 on real embeddings, in phase 2
    and again in phase 3. In the reference code this loss is only ever
    computed on real embeddings; generated sequences get S's learned dynamics
    because the generator's output is passed through S (h_hat = S(G(z))).
    This is the paper's answer to an adversarial loss alone not capturing
    step-wise conditional distributions.

    The reference code gives the supervisor one GRU layer fewer than the other
    networks (num_layer - 1), hence the default of 2 layers here.

    Args:
        n_tickers (int): number of ticker ids.
        latent_dim (int | None): K; must match the Embedder's latent_dim.
        cond_dim, hidden_dim: see ConditionalGRU.
        num_layers (int): GRU layers (default 2 = 3 - 1, as in the reference code).
    """

    def __init__(
        self,
        n_tickers,
        latent_dim=None,
        cond_dim=8,
        hidden_dim=24,
        num_layers=2,
    ):
        super().__init__()
        self.latent_dim = latent_dim or hidden_dim
        self.net = ConditionalGRU(
            self.latent_dim,
            self.latent_dim,
            n_tickers,
            cond_dim,
            hidden_dim,
            num_layers,
            activation=nn.Sigmoid(),
        )

    def forward(self, h, ticker_ids):
        """
        Args:
            h (torch.Tensor): (B, T, K) real or generated latent sequence.
            ticker_ids (torch.LongTensor): (B,) ticker ids.

        Returns:
            torch.Tensor (B, T, K); step t is the prediction of h[:, t + 1].
        """
        return self.net(h, ticker_ids)


class Discriminator(nn.Module):
    """
    TimeGAN discriminator D: one real/fake logit per step of a latent sequence.

    Identical in design to RCGANDiscriminator (conditional, bidirectional GRU,
    linear head, raw logits for BCEWithLogitsLoss), so the two models differ
    only in the components under test. The difference is the input: TimeGAN's
    adversarial game is played in the latent space (TimeGAN paper, Section
    4.2), so D receives K-dimensional latents (real embeddings E(x), or
    generator output with or without the Supervisor) instead of F features.
    Kept as a separate class so the two models' weights and defaults never mix.

    Args:
        n_tickers (int): number of ticker ids.
        latent_dim (int | None): K, input size; None uses hidden_dim, matching
            the Embedder's default.
        cond_dim, hidden_dim, num_layers: see ConditionalGRU.
    """

    def __init__(
        self,
        n_tickers,
        latent_dim=None,
        cond_dim=8,
        hidden_dim=24,
        num_layers=3,
    ):
        super().__init__()
        self.latent_dim = latent_dim or hidden_dim
        self.net = ConditionalGRU(
            self.latent_dim,
            1,
            n_tickers,
            cond_dim,
            hidden_dim,
            num_layers,
            bidirectional=True,
        )

    def forward(self, h, ticker_ids):
        """
        Args:
            h (torch.Tensor): (B, T, K) real or synthetic latent sequence.
            ticker_ids (torch.LongTensor): (B,) ticker ids.

        Returns:
            torch.Tensor (B, T) logits; > 0 means "looks real".
        """
        return self.net(h, ticker_ids).squeeze(-1)


# =============================================================================
# Utilities
# =============================================================================


def count_parameters(module):
    """Number of trainable parameters in a module (for the resource table)."""
    return sum(p.numel() for p in module.parameters() if p.requires_grad)

import numpy as np


class DataGenerator:
    def __init__(self, p, q, deg, eps_bar, seed=None):
        """
        Synthetic contextual coefficient generator following the repo's benchmark family.

        Args:
            p: Feature/context dimension
            q: Output dimension
            deg: Polynomial degree
            eps_bar: Noise level (0 for no noise, 0.5 for moderate noise)
            seed: Random seed
        """
        self.p = p
        self.q = q
        self.deg = deg
        self.eps_bar = eps_bar
        self.rng = np.random.RandomState(seed)
        self.W = self.rng.binomial(1, 0.5, size=(q, p)).astype(float)

    def generate_context(self):
        """Generate context x_t ~ N(0, I_p)."""
        return self.rng.randn(self.p)

    def get_latent_vec(self, x):
        """
        Generate the latent coefficient vector for a context.

        Formula: [1 + (1 + W_j^T x_t / sqrt(p))^deg] * epsilon_{t,j}
        where epsilon_{t,j} ~ Uniform[1-eps_bar, 1+eps_bar]
        """
        z = (self.W @ x) / np.sqrt(self.p)
        base = 1.0 + np.power(1.0 + z, self.deg)
        if self.eps_bar > 0:
            eps = self.rng.uniform(1.0 - self.eps_bar, 1.0 + self.eps_bar, size=self.q)
        else:
            eps = np.ones(self.q)
        return base * eps

    def get_reward(self, coeff_t, w):
        """Compute the realized linear objective coeff_t^T w."""
        return float(np.dot(coeff_t, w))


class ShortestPathCalibratedPolyGenerator:
    """Calibrated shortest-path polynomial generator.

    This family interpolates between the repo's earlier proxy and the original
    SPO generator:

        c_j(x) = (poly_offset + poly_scale * W_j^T x / sqrt(p))^deg + 1

    followed by multiplicative noise epsilon_j ~ Uniform[1-eps_bar, 1+eps_bar].
    """

    def __init__(self, p, q, deg, eps_bar, *, poly_offset, poly_scale, seed=None):
        self.p = int(p)
        self.q = int(q)
        self.deg = int(deg)
        self.eps_bar = float(eps_bar)
        self.poly_offset = float(poly_offset)
        self.poly_scale = float(poly_scale)
        self.rng = np.random.RandomState(seed)
        self.W = self.rng.binomial(1, 0.5, size=(self.q, self.p)).astype(float)

    def generate_context(self):
        """Generate context x_t ~ N(0, I_p)."""
        return self.rng.randn(self.p)

    def get_latent_vec(self, x):
        """Generate edge costs using the calibrated shortest-path polynomial map."""
        z = (self.W @ x) / np.sqrt(self.p)
        base = np.power(self.poly_offset + self.poly_scale * z, self.deg) + 1.0
        if self.eps_bar > 0:
            eps = self.rng.uniform(1.0 - self.eps_bar, 1.0 + self.eps_bar, size=self.q)
        else:
            eps = np.ones(self.q)
        return base * eps

    def get_reward(self, coeff_t, w):
        return float(np.dot(coeff_t, w))

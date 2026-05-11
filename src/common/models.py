"""
Reward prediction models for contextual optimization.

Supports:
- Linear models: μ̂(x) = M @ x
- Two-layer neural networks: μ̂(x) = W2 @ relu(W1 @ x + b1) + b2
- Shared local-feature scorers over context matrices
"""
import numpy as np


class RewardModel:
    """Base class for reward prediction models."""
    model_family = "point"
    uses_torch = False
    context_mode = "global"
    
    def __init__(self, p, q, seed=None):
        """
        Args:
            p: Input dimension (benchmark-defined context features)
            q: Output dimension (decision space)
            seed: Random seed for initialization
        """
        self.p = p
        self.q = q
        self.rng = np.random.RandomState(seed)
        
    def predict(self, x, theta):
        """
        Predict reward coefficients.
        
        Args:
            x: Benchmark-defined context input
            theta: Model parameters (flat array)
            
        Returns:
            mu_hat: Predicted coefficients (q,)
        """
        raise NotImplementedError
        
    def compute_grad_r(self, x, w, theta):
        """
        Compute gradient of predicted scalar reward r_hat = w^T * mu_hat(x)
        with respect to theta.
        
        Args:
            x: Benchmark-defined context input
            w: Weight vector (q,)
            theta: Model parameters (flat array)
            
        Returns:
            grad: Gradient vector (same shape as theta)
        """
        raise NotImplementedError
        
    def get_param_dim(self):
        """Return total number of parameters."""
        raise NotImplementedError
        
    def initialize_params(self):
        """Return initial parameters."""
        raise NotImplementedError


class LinearModel(RewardModel):
    """
    Linear reward model: μ̂(x; θ) = M @ x
    
    Parameters: θ = vec(M) where M is (q x p) matrix
    """
    
    def get_param_dim(self):
        return self.q * self.p
    
    def initialize_params(self):
        """Small random initialization."""
        return self.rng.randn(self.get_param_dim()) * 0.01
    
    def predict(self, x, theta):
        """Linear prediction: M @ x"""
        M = theta.reshape((self.q, self.p))
        return M @ x
        
    def compute_grad_r(self, x, w, theta):
        """
        Gradient for Linear Model.
        r_hat = w^T M x = sum_{i,j} w_i M_{ij} x_j
        d(r_hat)/d(M_{ij}) = w_i * x_j
        """
        # Outer product w * x^T flattened
        # Corresponds to vec(w x^T) if theta is vec(M) row-major
        grad_M = np.outer(w, x).flatten()
        return grad_M


class TwoLayerNN(RewardModel):
    """
    Two-layer neural network: μ̂(x; θ) = W2 @ relu(W1 @ x + b1) + b2
    
    Architecture:
    - Input: p dimensions
    - Hidden: 128 neurons with ReLU activation
    - Output: q dimensions
    
    Parameters: θ = [vec(W1), vec(b1), vec(W2), vec(b2)]
    - W1: (128 x p)
    - b1: (128,)
    - W2: (q x 128)
    - b2: (q,)
    """
    
    def __init__(self, p, q, hidden_dim=128, seed=None):
        super().__init__(p, q, seed)
        self.hidden_dim = hidden_dim
        
        # Parameter dimensions
        self.w1_size = hidden_dim * p
        self.b1_size = hidden_dim
        self.w2_size = q * hidden_dim
        self.b2_size = q
        
    def get_param_dim(self):
        return self.w1_size + self.b1_size + self.w2_size + self.b2_size
    
    def initialize_params(self):
        """Xavier/He initialization for neural network."""
        # W1: Xavier initialization
        w1 = self.rng.randn(self.w1_size) * np.sqrt(2.0 / self.p)
        b1 = np.zeros(self.b1_size)
        
        # W2: Xavier initialization
        w2 = self.rng.randn(self.w2_size) * np.sqrt(2.0 / self.hidden_dim)
        b2 = np.zeros(self.b2_size)
        
        return np.concatenate([w1, b1, w2, b2])
    
    def predict(self, x, theta):
        """
        Forward pass: W2 @ relu(W1 @ x + b1) + b2
        
        Args:
            x: Context (p,)
            theta: Flat parameter vector
            
        Returns:
            mu_hat: Predicted coefficients (q,)
        """
        # Unpack parameters
        offset = 0
        
        w1 = theta[offset:offset + self.w1_size].reshape(self.hidden_dim, self.p)
        offset += self.w1_size
        
        b1 = theta[offset:offset + self.b1_size]
        offset += self.b1_size
        
        w2 = theta[offset:offset + self.w2_size].reshape(self.q, self.hidden_dim)
        offset += self.w2_size
        
        b2 = theta[offset:offset + self.b2_size]
        
        # Forward pass
        hidden = w1 @ x + b1
        hidden = np.maximum(0, hidden)  # ReLU
        output = w2 @ hidden + b2
        
        return output

    def compute_grad_r(self, x, w, theta):
        """
        Gradient for TwoLayerNN.
        r_hat = w^T (W2 relu(W1 x + b1) + b2)
        """
        # Unpack parameters (need them for forward pass state)
        offset = 0
        w1 = theta[offset:offset + self.w1_size].reshape(self.hidden_dim, self.p)
        offset += self.w1_size
        b1 = theta[offset:offset + self.b1_size]
        offset += self.b1_size
        w2 = theta[offset:offset + self.w2_size].reshape(self.q, self.hidden_dim)
        offset += self.w2_size
        b2 = theta[offset:offset + self.b2_size]
        
        # Forward pass
        z1 = w1 @ x + b1
        h1 = np.maximum(0, z1)
        # out = w2 @ h1 + b2  (not needed explicitly, we need gradients)
        
        # Backward pass for r_hat = w^T out
        # d(r_hat)/d(out) = w
        grad_out = w
        
        # d(r_hat)/d(b2) = grad_out = w
        grad_b2 = grad_out
        
        # d(r_hat)/d(W2) = grad_out @ h1^T
        grad_w2 = np.outer(grad_out, h1).flatten()
        
        # d(r_hat)/d(h1) = W2^T @ grad_out
        grad_h1 = w2.T @ grad_out
        
        # d(r_hat)/d(z1) = grad_h1 * (z1 > 0)
        grad_z1 = grad_h1 * (z1 > 0)
        
        # d(r_hat)/d(b1) = grad_z1
        grad_b1 = grad_z1
        
        # d(r_hat)/d(W1) = grad_z1 @ x^T
        grad_w1 = np.outer(grad_z1, x).flatten()
        
        return np.concatenate([grad_w1, grad_b1, grad_w2, grad_b2])


def _validate_local_feature_context(x, q, p):
    X = np.asarray(x, dtype=float)
    if X.ndim != 2:
        raise ValueError(f"Expected local-feature context with shape ({q}, {p}), got ndim={X.ndim}")
    if X.shape != (q, p):
        raise ValueError(f"Expected local-feature context shape ({q}, {p}), got {X.shape}")
    if not np.all(np.isfinite(X)):
        raise ValueError("Context contains non-finite values")
    return X


def _validate_weight_vector(w, q):
    weights = np.asarray(w, dtype=float).reshape(-1)
    if weights.shape != (q,):
        raise ValueError(f"Expected weight vector shape ({q},), got {weights.shape}")
    if not np.all(np.isfinite(weights)):
        raise ValueError("Weight vector contains non-finite values")
    return weights


class SharedLinearModel(RewardModel):
    """Shared linear scorer applied row-wise to a local-feature context matrix."""
    context_mode = "shared_local"

    def get_param_dim(self):
        return self.p

    def initialize_params(self):
        return self.rng.randn(self.get_param_dim()) * 0.01

    def predict(self, x, theta):
        X = _validate_local_feature_context(x, self.q, self.p)
        params = np.asarray(theta, dtype=float).reshape(self.p)
        return X @ params

    def compute_grad_r(self, x, w, theta):
        del theta
        X = _validate_local_feature_context(x, self.q, self.p)
        weights = _validate_weight_vector(w, self.q)
        return X.T @ weights


class SharedTwoLayerNN(RewardModel):
    """Shared one-hidden-layer MLP applied row-wise to a local-feature context matrix."""
    context_mode = "shared_local"

    def __init__(self, p, q, hidden_dim=128, seed=None):
        super().__init__(p, q, seed)
        self.hidden_dim = hidden_dim
        self.w1_size = hidden_dim * p
        self.b1_size = hidden_dim
        self.w2_size = hidden_dim
        self.b2_size = 1

    def get_param_dim(self):
        return self.w1_size + self.b1_size + self.w2_size + self.b2_size

    def initialize_params(self):
        w1 = self.rng.randn(self.w1_size) * np.sqrt(2.0 / self.p)
        b1 = np.zeros(self.b1_size)
        w2 = self.rng.randn(self.w2_size) * np.sqrt(2.0 / self.hidden_dim)
        b2 = np.zeros(self.b2_size)
        return np.concatenate([w1, b1, w2, b2])

    def _unpack(self, theta):
        offset = 0
        w1 = theta[offset:offset + self.w1_size].reshape(self.hidden_dim, self.p)
        offset += self.w1_size
        b1 = theta[offset:offset + self.b1_size]
        offset += self.b1_size
        w2 = theta[offset:offset + self.w2_size]
        offset += self.w2_size
        b2 = float(theta[offset])
        return w1, b1, w2, b2

    def predict(self, x, theta):
        X = _validate_local_feature_context(x, self.q, self.p)
        w1, b1, w2, b2 = self._unpack(np.asarray(theta, dtype=float).reshape(-1))
        z1 = X @ w1.T + b1
        h1 = np.maximum(0.0, z1)
        return h1 @ w2 + b2

    def compute_grad_r(self, x, w, theta):
        X = _validate_local_feature_context(x, self.q, self.p)
        weights = _validate_weight_vector(w, self.q)
        w1, b1, w2, _ = self._unpack(np.asarray(theta, dtype=float).reshape(-1))

        z1 = X @ w1.T + b1
        h1 = np.maximum(0.0, z1)

        grad_b2 = np.array([np.sum(weights)], dtype=float)
        grad_w2 = h1.T @ weights
        grad_h1 = np.outer(weights, w2)
        grad_z1 = grad_h1 * (z1 > 0.0)
        grad_b1 = np.sum(grad_z1, axis=0)
        grad_w1 = grad_z1.T @ X

        return np.concatenate([grad_w1.flatten(), grad_b1, grad_w2, grad_b2])


def _validate_benchmark_model_choice(model_type, config):
    model_type = _canonical_model_type(model_type)
    benchmark = str((config or {}).get("benchmark", "")).lower()
    allowed_shared = {"shared_linear", "shared_nn", "shared_diffusion", "shared_cnf"}
    if benchmark in {"energy", "pricing"} and model_type not in allowed_shared:
        raise ValueError(
            f"benchmark='{benchmark}' requires a shared local-feature model_type in {sorted(allowed_shared)}; "
            f"got '{model_type}'."
        )


def _point_model_hidden_dim(config):
    """Hidden width for point-model MLPs."""
    if config is None:
        return 128
    return int(config.get("nn_hidden_dim", 128))


def _canonical_model_type(model_type):
    return str(model_type).lower()


def create_model(model_type, p, q, seed=None, config=None):
    """
    Factory function to create reward models.
    
    Args:
        model_type: 'linear', 'nn', 'shared_linear', 'shared_nn', 'diffusion',
            'cnf', 'shared_diffusion', or 'shared_cnf'
        p: Input dimension
        q: Output dimension
        seed: Random seed
        config: Optional full experiment config (required by generative models)
        
    Returns:
        RewardModel instance
    """
    _validate_benchmark_model_choice(model_type, config)
    model_type = _canonical_model_type(model_type)

    if model_type == 'linear':
        return LinearModel(p, q, seed)
    elif model_type == 'nn':
        return TwoLayerNN(p, q, hidden_dim=_point_model_hidden_dim(config), seed=seed)
    elif model_type == 'shared_linear':
        return SharedLinearModel(p, q, seed=seed)
    elif model_type == 'shared_nn':
        return SharedTwoLayerNN(p, q, hidden_dim=_point_model_hidden_dim(config), seed=seed)
    elif model_type in {'diffusion', 'cnf', 'shared_diffusion', 'shared_cnf'}:
        from src.common.generative_models import (
            ConditionalRealNVPModel,
            DiffusionGenerativeModel,
            SharedDiffusionGenerativeModel,
            SharedRealNVPModel,
        )

        if model_type == 'diffusion':
            return DiffusionGenerativeModel(p, q, seed=seed, config=config)
        if model_type == 'shared_diffusion':
            return SharedDiffusionGenerativeModel(p, q, seed=seed, config=config)
        if model_type == 'shared_cnf':
            return SharedRealNVPModel(p, q, seed=seed, config=config)
        return ConditionalRealNVPModel(p, q, seed=seed, config=config)
    else:
        raise ValueError(
            "Unknown model type: "
            f"{model_type}. Choose 'linear', 'nn', 'shared_linear', 'shared_nn', "
            "'diffusion', 'cnf', 'shared_diffusion', or 'shared_cnf'."
        )

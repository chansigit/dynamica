from flax import linen as nn
import jax.numpy as jnp
import jax

def softmax_one(x, axis=-1):
    x = x - jnp.max(x, axis=axis, keepdims=True)
    exp_x = jnp.exp(x)
    return exp_x / (1 + jnp.sum(exp_x, axis=axis, keepdims=True))

class SpatialAttentionLayer(nn.Module):
    input_dim: int
    hidden_dim: int
    output_dim: int = None
    learnable_sigma: bool = False
    sigma_init: float = 1.0
    p_dim: int = 3
    residual: bool = True
    message_passing: bool = True  # 🆕 New flag

    def setup(self):
        output_dim = self.output_dim or self.hidden_dim
        self.W_q = nn.Dense(self.hidden_dim)
        self.W_k = nn.Dense(self.hidden_dim)
        self.W_v = nn.Dense(output_dim)

        if self.learnable_sigma:
            self.log_sigma = self.param(
                "log_sigma", lambda key: jnp.log(jnp.array(self.sigma_init))
            )
        else:
            self.sigma = self.sigma_init

    def __call__(self, X, P=None, return_attention=False):
        if P is None:
            P = X[:, -self.p_dim:]
            X = X[:, :-self.p_dim]

        Q = self.W_q(X)
        K = self.W_k(X)
        V = self.W_v(X)

        if self.message_passing:
            d = self.hidden_dim
            scores = Q @ K.T / jnp.sqrt(d)
            dist2 = jnp.sum((P[:, None, :] - P[None, :, :]) ** 2, axis=-1)
            sigma2 = (
                jnp.exp(self.log_sigma) ** 2
                if self.learnable_sigma
                else self.sigma ** 2
            )
            spatial_decay = jnp.exp(-dist2 / (2 * sigma2))
            scores = scores * spatial_decay
            A = softmax_one(scores, axis=-1)
            H = A @ V
        else:
            H = V
            A = None

        if self.residual and H.shape == X.shape:
            H += X

        if return_attention:
            return H, A
        return H

    def get_attention_matrix(self, variables, X, P=None):
        _, A = self.apply(variables, X, P, return_attention=True)
        return A
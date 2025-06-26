from flax import linen as nn
import jax.numpy as jnp
import emlp.nn.flax as emlp
from emlp.groups import SO
from emlp.reps import T, V
from .attention import SpatialAttentionLayer

class DynVelocity(nn.Module):
    feature_dim: int = 30
    position_dim: int = 3
    hidden_dim: int = 32
    output_dim: int = 30
    center_positions: bool = True
    static_pos: bool = False
    message_passing: bool = True

    def setup(self):
        self.backbone = nn.Sequential([
            SpatialAttentionLayer(
                input_dim=self.feature_dim,
                hidden_dim=self.hidden_dim,
                output_dim=self.output_dim,
                message_passing=self.message_passing
            ),
            nn.LayerNorm(),
            nn.Dense(128),
            nn.silu,
            nn.Dense(self.output_dim)
        ])

        self.expr_head = nn.Dense(self.feature_dim)

        G = SO(3)
        repin = sum([T(0)] * self.feature_dim) + V
        repout = V
        self.velocity_head = emlp.EMLP(repin, repout, G)

    def __call__(self, X):
        P = X[:, -self.position_dim:]
        X_expr = X[:, :-self.position_dim]

        if self.center_positions:
            P = P - jnp.mean(P, axis=0, keepdims=True)

        H = self.backbone(X)
        feature_velo = self.expr_head(H)

        if self.static_pos:
            pos_velo = jnp.zeros_like(P)
        else:
            emlp_input = jnp.concatenate([X_expr, P], axis=-1)
            pos_velo = self.velocity_head(emlp_input)

        return feature_velo, pos_velo
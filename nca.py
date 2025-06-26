import torch
import torch.nn as nn
from .sat import SpatialAttentionLayer
from .equi import E3NNVelocityPredictor

class DynVelocity(nn.Module):
    """
    Dynamic velocity prediction model for simultaneously predicting velocities in feature space and physical space.
    
    This model combines spatial attention mechanisms with equivariant neural networks to process data with spatial structure,
    predicting system evolution while maintaining physical consistency.
    
    Args:
        input_dim (int): Dimension of input features. Default: 30
        position_dim (int): Dimension of position vectors. Default: 3 (3D space)
        hidden_dim (int): Dimension of hidden layers. Default: 32
        output_dim (int): Dimension of output features. Default: 30
        sigma (float): Sigma parameter for spatial attention layer, controlling spatial decay. Default: 0.3
        static_pos (bool): Whether to use static positions (no position velocity prediction). Default: True
        message_passing (bool): Whether to use message passing mechanism. Default: True
        autonomous (bool): Whether the system is autonomous (time-independent). Default: False
    
    Attributes:
        backbone: Network that transforms input features and positions into latent representations
        expr_head: Network that predicts velocities in feature space
        velocity_head: Equivariant network that predicts velocities in physical space
    """
    def __init__(
        self,
        input_dim=30,
        position_dim=3,
        hidden_dim=32,
        output_dim=30,
        sigma = 0.3,
        static_pos=True,
        message_passing=True,
        autonomous = False,
        kinetic_energy = True
    ):
        super().__init__()
        self.input_dim = input_dim
        self.position_dim = position_dim
        self.static_pos = static_pos
        self.autonomous = autonomous
        
        # Expression + position --> latent representation
        self.backbone = nn.Sequential(
            SpatialAttentionLayer(
                input_dim = input_dim, p_dim = position_dim,
                hidden_dim = hidden_dim, output_dim = output_dim, residual = False,
                message_passing = message_passing, sigma = sigma, use_softmax = False),
            nn.LayerNorm(output_dim), nn.Linear(output_dim, 128), nn.SiLU(),
            nn.Linear(128, output_dim)
        )

        # Predict velocities for expressions (latent space) 
        self.expr_head = nn.Linear(output_dim, output_dim)

        # Predict velocities for physical positions
        n_scalars = output_dim if autonomous else output_dim + 1
        self.velocity_head = E3NNVelocityPredictor(n_scalars=n_scalars, n_vec3d=1, scalar_hidden=128, vec3d_hidden=128, n_vec3d_out=1)

        self.kinetic_energy = kinetic_energy


    def forward(self, t, Z, args=None):
        """
        Z: Tensor of shape (N, input_dim + position_dim + energy_dim), energy_dim is 1 if any.
        Returns:
            Concatenated tensor of:
                feature_velo: Expression velocities (N, input_dim)
                pos_velo: Position velocities (N, position_dim)
        """
        # Split expression and position
        E = Z[:, [-1]]
        X = Z[:, :-1 ]
        P = X[:, -self.position_dim:]
        H = self.backbone(X)
        
        # Predict velocities for expressions (in latent space)
        feature_velo = self.expr_head(H)
        
        # Add time information for non-autonomous systems
        if not self.autonomous:
            H = torch.cat([H, t.expand(H.size(0), 1)], dim=1)
            
        # Predict position velocities (zero if static)
        if self.static_pos:
            pos_velo = torch.zeros_like(P)
        else:
            pos_velo = self.velocity_head.forward(
                self.velocity_head.prepare_input(H, P.reshape(-1, 3))
            )
        if not self.kinetic_energy:
            return torch.concat([feature_velo, pos_velo], axis=1)
        else:
            
            total_kinetic_energy = 0.5*(feature_velo **2).mean(axis=1) + 0.5*(pos_velo **2).mean(axis=1)
            return torch.concat([feature_velo, pos_velo, total_kinetic_energy.unsqueeze(1)], axis=1)

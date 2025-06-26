import torch
import torch.nn as nn
#from emlp.nn.pytorch import EMLP
#from emlp.reps import T, V
#from emlp.groups import SO
from .sa import SpatialAttentionLayer
import time

class DynVelocity(nn.Module):
    def __init__(
        self,
        feature_dim=30,
        position_dim=3,
        hidden_dim=32,
        output_dim=30,
        learnable_sigma=False,
        sigma_init = 0.3,
        static_pos=False,
        message_passing=True,
        use_emlp=True
    ):
        super().__init__()

        self.feature_dim = feature_dim
        self.position_dim = position_dim
        self.static_pos = static_pos
        self.use_emlp = use_emlp

        # Expression + position --> latent representation
        self.backbone = nn.Sequential(
            SpatialAttentionLayer(
                input_dim=feature_dim,
                hidden_dim=hidden_dim,
                output_dim=output_dim,
                p_dim = position_dim,
                message_passing = message_passing,
                use_softmax = False,
                learnable_sigma = learnable_sigma, 
                sigma_init=sigma_init
            ),
            nn.LayerNorm(output_dim), nn.Linear(output_dim, 128), nn.SiLU(),
            nn.Linear(128, output_dim)
        )

        # Predict expression velocity
        self.expr_head = nn.Sequential(
            nn.Linear(output_dim, feature_dim)#, nn.LayerNorm(output_dim), nn.SiLU()
        )

        # Predict position velocity
        if self.use_emlp:
            G = SO(3)
            repin = sum([T(0)] * output_dim) + V
            self.velocity_head = EMLP(repin, V, G)
        else:
            self.velocity_head = nn.Sequential(
                nn.Linear(output_dim + position_dim, 64),
                nn.ReLU(),
                nn.Linear(64, position_dim)
            )

    def forward(self, t, X, args=None):
        """
        X: Tensor of shape (N, feature_dim + position_dim)
        Returns:
            feature_velo: (N, feature_dim)
            pos_velo:     (N, position_dim)
        """
        # Split expression and position
        P = X[:, -self.position_dim:]
        X_expr = X[:, :-self.position_dim]

        #t1 = time.time()
        H = self.backbone(torch.cat([X_expr, P], dim=-1))
        
        feature_velo = self.expr_head(H)
        #print( time.time()-t1,' time for expr' )

        #t1 = time.time()
        if self.static_pos:
            pos_velo = torch.zeros_like(P)
        else:
            pos_input = torch.cat([H, P], dim=-1)
            pos_velo = self.velocity_head(pos_input)
        #print( time.time()-t1,' time for pos' )
        
        return torch.concat([feature_velo, pos_velo], axis=1)

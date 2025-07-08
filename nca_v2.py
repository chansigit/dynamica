import torch
import torch.nn as nn
import torch.nn.init as init
#from dynamica.sat import SpatialAttentionLayer
#from dynamica.equi import E3NNVelocityPredictor
from .sat import SpatialAttentionLayer
from .equi import E3NNVelocityPredictor




class DynVelocity(nn.Module):
    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            init.kaiming_normal_(m.weight, nonlinearity='relu')
            if m.bias is not None:
                init.zeros_(m.bias)
        elif isinstance(m, nn.Conv2d):
            init.kaiming_normal_(m.weight, nonlinearity='relu')
            if m.bias is not None:
                init.zeros_(m.bias)
        elif isinstance(m, nn.BatchNorm1d) or isinstance(m, nn.BatchNorm2d):
            init.ones_(m.weight)
            init.zeros_(m.bias)
    
    def __init__(
        self,
        input_dim = 30,
        output_dim=30,
        hidden_dim = 128,
        position_dim = 3,
        sigma =0.3,
        static_pos=True,
        message_passing = True,
        expr_autonomous = True,
        pos_autonomous = False,
        energy_regularization = True
    ):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.position_dim = position_dim
        self.sigma = sigma
        self.message_passing = message_passing
        self.static_pos = static_pos
        self.energy_regularization = energy_regularization
        self.expr_autonomous = expr_autonomous
        self.pos_autonomous  = pos_autonomous

        self.mlp_base = nn.Sequential(
            nn.Linear(input_dim+position_dim, hidden_dim), nn.LayerNorm(hidden_dim), nn.SiLU(), 
            nn.Linear(hidden_dim, hidden_dim), nn.LayerNorm(hidden_dim), nn.SiLU(), 
            nn.Linear(hidden_dim, hidden_dim)
        )
        self.spatial_base1 = nn.Sequential(
            SpatialAttentionLayer(
                input_dim = input_dim, p_dim = position_dim,
                hidden_dim = hidden_dim, output_dim = hidden_dim, 
                residual = False, message_passing = True, sigma = self.sigma, use_softmax = False),
            nn.Linear(hidden_dim, hidden_dim), nn.LayerNorm(hidden_dim),  nn.SiLU(),
            #nn.Linear(hidden_dim, hidden_dim), nn.LayerNorm(hidden_dim),  
        )
        self.spatial_base2 = nn.Sequential(
            SpatialAttentionLayer(
                input_dim = hidden_dim, p_dim = position_dim,
                hidden_dim = hidden_dim, output_dim = hidden_dim, 
                residual = False, message_passing = True, sigma = self.sigma, use_softmax = False),
            nn.Linear(hidden_dim, hidden_dim), nn.LayerNorm(hidden_dim),  nn.SiLU()
        )

        expr_head_indim = hidden_dim*2 if self.expr_autonomous else hidden_dim*2 + 1
        self.expr_head = nn.Sequential(
            nn.Linear(expr_head_indim, hidden_dim), nn.LayerNorm(hidden_dim), nn.SiLU(), nn.Linear(hidden_dim, output_dim)
        )
        
        # Predict velocities for physical positions
        n_scalars = hidden_dim*2 if self.pos_autonomous else hidden_dim*2 + 1
        
        self.pos_head = E3NNVelocityPredictor(n_scalars=n_scalars, n_vec3d=1, scalar_hidden=128, vec3d_hidden=128, n_vec3d_out=1)

        self.apply(self._init_weights)

    def forward(self, t, Z, args=None):
        # Split expression and position
        E = Z[:, [-1]]
        X = Z[:, :-1 ]
        P = X[:, -self.position_dim:]


        # calculate velocity representation in stand alone mode and spatial aware mode, and then aggregate 
        H0 = self.mlp_base(X)
        
        if self.message_passing:
            H1 = self.spatial_base1(X)
            H1 = self.spatial_base2(torch.cat([H1, P], 1))
        else:
            H1 = torch.zeros_like(H0)
        H = torch.concat([H0 , H1], axis = 1)

        # Predict velocities for expressions (in latent space)

        if not self.expr_autonomous:
            H_e = torch.cat([H, t.expand(H.size(0), 1)], dim=1)
            feature_velo = self.expr_head(H_e)
        else:
            feature_velo = self.expr_head(H)

        if self.static_pos:
            pos_velo = torch.zeros_like(P)
        else:
            if not self.pos_autonomous:
                H_p = torch.cat([H, t.expand(H.size(0), 1)], dim=1)
                pos_velo = self.pos_head.forward(
                    self.pos_head.prepare_input(H_p, P.reshape(-1, 3))
                )
            else:
                pos_velo = self.pos_head.forward(
                    self.pos_head.prepare_input(H, P.reshape(-1, 3))
                )

        # concatenate velocities
        dynamics = torch.concat([feature_velo, pos_velo], axis=1)
        
        # calculate kinetic energy
        if self.energy_regularization:
            total_kinetic_energy = 0.5*(feature_velo **2).mean(axis=1) + 0.5*(pos_velo **2).mean(axis=1)
            dynamics = torch.concat([ dynamics, total_kinetic_energy.unsqueeze(1)], axis=1)
            
        return dynamics
        

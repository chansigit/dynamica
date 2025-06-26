import torch
from e3nn.o3 import Irreps, Linear
from e3nn.nn import Activation
#from e3nn.o3 import rand_matrix

class E3NNVelocityPredictor(torch.nn.Module):
    """Equivariant neural network for predicting 3D vector quantities (e.g., velocities).
    
    This model preserves rotational equivariance, meaning that rotating the input
    will result in the same rotation being applied to the output. This is achieved
    using the E3NN framework which implements SO(3) equivariant neural networks.
    
    The network processes both scalar (invariant) features and vector (equivariant)
    features. Scalar features remain unchanged under rotation, while vector features
    transform appropriately.
    
    Args:
        n_scalars: Number of scalar (rotation invariant) input features
        n_vec3d: Number of 3D vector (rotation equivariant) input features
        scalar_hidden: Number of scalar features in hidden layers
        vec3d_hidden: Number of vector features in hidden layers
        n_vec3d_out: Number of output 3D vector features
        
    Attributes:
        irreps_in: Input irreducible representations
        irreps_out: Output irreducible representations
        net: The equivariant neural network
    """
    def __init__(
        self,
        n_scalars: int,
        n_vec3d: int,
        scalar_hidden: int = 8,
        vec3d_hidden: int = 8,
        n_vec3d_out: int = 1,
    ):
        super().__init__()

        self.n_scalars = n_scalars
        self.n_vec3d = n_vec3d

        self.irreps_in  = Irreps(f"{n_scalars}x0e + {n_vec3d}x1o")
        self.irreps_out = Irreps(f"{n_vec3d_out}x1o")
        hidden_irreps   = Irreps(f"{scalar_hidden}x0e + {vec3d_hidden}x1o")

        # Apply ReLU only to scalars (l=0), no activation for vectors (l=1) to preserve equivariance
        acts = [torch.relu if ir.l == 0 else None for mul, ir in hidden_irreps]

        self.net = torch.nn.Sequential(
            Linear(self.irreps_in, hidden_irreps), Activation(hidden_irreps, acts=acts),
            Linear(hidden_irreps, hidden_irreps),  Activation(hidden_irreps, acts=acts),
            Linear(hidden_irreps, self.irreps_out),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the network.
        
        Args:
            x: Input tensor containing both scalar and vector features,
               formatted as expected by E3NN (use prepare_input to format correctly)
               
        Returns:
            Tensor of equivariant vector outputs
        """
        return self.net(x)

    def prepare_input(self, invariant_feat: torch.Tensor, equivariant_feat: torch.Tensor) -> torch.Tensor:
        """Prepare input by concatenating scalar and vector features.
        
        This method ensures the input is properly formatted for the E3NN model.
        
        Args:
            invariant_feat: Tensor of scalar (invariant) features with shape [..., n_scalars]
            equivariant_feat: Tensor of vector (equivariant) features with shape [..., n_vec3d*3]
                              where each vector is flattened (x1,y1,z1,x2,y2,z2,...)
                              
        Returns:
            Properly formatted input tensor for the E3NN model
            
        Raises:
            AssertionError: If input dimensions don't match expected dimensions
        """
        assert invariant_feat.shape[-1] == self.n_scalars
        assert equivariant_feat.shape[-1] == self.n_vec3d * 3
        return torch.cat([invariant_feat, equivariant_feat], dim=-1)

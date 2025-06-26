import torch
import torch.nn as nn
import torch.nn.functional as F

def softmax_one(x, dim=None):
    x = x - x.max(dim=dim, keepdim=True).values
    exp_x = torch.exp(x)
    return exp_x / (1 + exp_x.sum(dim=dim, keepdim=True))

class SpatialAttentionLayer(nn.Module):
    """Spatial Attention Layer that incorporates spatial information into the attention mechanism.
    
    Mathematical formulation:
    1. Attention score calculation:
       scores = (W_Q·X)(W_K·X)^T / sqrt(d)
       where d is the hidden dimension, X is the input features
    
    2. Spatial decay:
       spatial_decay = exp(-||P_i - P_j||^2 / (2σ^2))
       where P_i, P_j are spatial coordinates, σ is the Gaussian kernel parameter
    
    3. Final attention weights:
       A = softmax(scores * spatial_decay)  or
       A = softmax_one(scores * spatial_decay)
    
    4. Output calculation:
       H = A·(W_V·X)
       If residual is enabled: H = H + X (or H + W_res·X if dimensions don't match)
    """
    def __init__(
        self,
        input_dim,
        hidden_dim,
        output_dim=None,
        use_softmax=False,
        sigma=1.0,
        residual=True,
        message_passing=True,
        p_dim=3,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim or hidden_dim
        self.residual = residual
        self.use_softmax = use_softmax
        self.message_passing = message_passing
        self.p_dim = p_dim
        self.sigma = sigma

        self.W_Q = nn.Linear(input_dim, hidden_dim)
        self.W_K = nn.Linear(input_dim, hidden_dim)
        self.W_V = nn.Linear(input_dim, self.output_dim)
        
        self.res_proj = nn.Linear(input_dim, self.output_dim) if (self.residual and input_dim != self.output_dim) else None

    def _compute_attention(self, X, P):
        """Compute attention weights given features X and positions P"""
        Q, K = self.W_Q(X), self.W_K(X)
        
        # Compute attention scores with spatial modulation
        scores = (Q @ K.T) / self.hidden_dim ** 0.5
        spatial_decay = torch.exp(-torch.cdist(P, P).pow(2) / (2 * self.sigma**2))
        scores = scores * spatial_decay
        
        # Normalize attention weights
        return F.softmax(scores, dim=-1) if self.use_softmax else softmax_one(scores, dim=-1)

    def forward(self, X):
        """Forward pass
        
        Args:
            X: Input tensor of shape [batch_size, feature_dim + p_dim]
               where the last p_dim dimensions are spatial coordinates
        
        Returns:
            H: Output features of shape [batch_size, output_dim]
        """
        P, X = X[:, -self.p_dim:], X[:, :-self.p_dim]
        
        if not self.message_passing:
            H = self.W_V(X)
        else:
            A = self._compute_attention(X, P)
            H = A @ self.W_V(X)
        
        # Apply residual connection if enabled
        if self.residual:
            H = H + (self.res_proj(X) if self.res_proj else X)
            
        return H

    def get_attention_matrix(self, X):
        """Get the attention matrix
        
        Args:
            X: Input tensor of shape [batch_size, feature_dim + p_dim]
        
        Returns:
            A: Attention matrix of shape [batch_size, batch_size]
               Returns None if message_passing=False
        """
        if not self.message_passing:
            return None
            
        P, X = X[:, -self.p_dim:], X[:, :-self.p_dim]
        return self._compute_attention(X, P)

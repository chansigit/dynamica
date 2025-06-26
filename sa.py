import torch
import torch.nn as nn
import torch.nn.functional as F

def softmax_one(x, dim=None):
    x = x - x.max(dim=dim, keepdim=True).values
    exp_x = torch.exp(x)
    return exp_x / (1 + exp_x.sum(dim=dim, keepdim=True))

class SpatialAttentionLayer(nn.Module):
    def __init__(
        self,
        input_dim,
        hidden_dim,
        output_dim=None,
        use_softmax=False,
        learnable_sigma=True,
        sigma_init=1.0,
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

        self.W_Q = nn.Linear(input_dim, hidden_dim)
        self.W_K = nn.Linear(input_dim, hidden_dim)
        self.W_V = nn.Linear(input_dim, self.output_dim)

        if learnable_sigma:
            self.log_sigma = nn.Parameter(torch.tensor([sigma_init]).log())
        else:
            self.register_buffer("sigma", torch.tensor([sigma_init]))

        if self.residual and self.input_dim != self.output_dim:
            self.res_proj = nn.Linear(self.input_dim, self.output_dim)
        else:
            self.res_proj = None

    def forward(self, X):
        P = X[:, -self.p_dim:]
        X = X[:, :-self.p_dim]

        Q = self.W_Q(X)
        K = self.W_K(X)
        V = self.W_V(X)

        A = None  # attention matrix placeholder

        if not self.message_passing:
            H = V
        else:
            scores = Q @ K.T
            scores = scores / self.hidden_dim ** 0.5

            dist2 = torch.cdist(P, P).pow(2)
            if hasattr(self, 'log_sigma'):
                sigma2 = torch.clamp(self.log_sigma.exp().pow(2), min=1e-4)
            else:
                sigma2 = self.sigma.pow(2)
            spatial_decay = torch.exp(-dist2 / (2 * sigma2))

            scores = scores * spatial_decay

            if self.use_softmax:
                A = F.softmax(scores, dim=-1)
            else:
                A = softmax_one(scores, dim=-1)

            H = A @ V

        if self.residual:
            if self.res_proj is not None:
                H = H + self.res_proj(X)
            else:
                H = H + X

        return H

    def get_attention_matrix(self, X):
        """get attention
        """
        if not self.message_passing:
            return None

        P = X[:, -self.p_dim:]
        X = X[:, :-self.p_dim]

        Q = self.W_Q(X)
        K = self.W_K(X)

        scores = Q @ K.T
        scores = scores / self.hidden_dim ** 0.5

        dist2 = torch.cdist(P, P).pow(2)
        if hasattr(self, 'log_sigma'):
            sigma2 = torch.clamp(self.log_sigma.exp().pow(2), min=1e-4)
        else:
            sigma2 = self.sigma.pow(2)
        spatial_decay = torch.exp(-dist2 / (2 * sigma2))

        scores = scores * spatial_decay

        if self.use_softmax:
            A = F.softmax(scores, dim=-1)
        else:
            A = softmax_one(scores, dim=-1)

        return A

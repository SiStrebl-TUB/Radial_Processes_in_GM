


import torch
import torch.nn as nn


#  Define models

### 2.2. Define MLP
class Swish(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return torch.sigmoid(x)*x
    

class NormalizeLogRadius(nn.Module):
    """Non-learnable preprocessing layer:
       x ↦ (x/||x||, log||x||).
    """
    def __init__(self, eps = 1e-6): 
        super().__init__()
        self.eps = eps

    def forward(self, x):
        # x: (batch, d)
        norm = torch.norm(x, dim=-1, keepdim=True)
        norm = norm + self.eps
        x_normalized = x / norm
        log_norm = torch.log(norm)
        return torch.cat([x_normalized, log_norm], dim=-1)
    
class PolarCoordinatesWithLogRadius(nn.Module):
    """Convert x ∈ R^d to polar/spherical coordinates.
       - d=2 → (r, θ)
       - d=3 → (r, θ, φ)
       Returns a tensor of size (..., d)
    """
    def __init__(self, eps=1e-12):
        super().__init__()
        self.eps = eps

    def forward(self, x):
        # x: (..., d)
        d = x.shape[-1]
        norm = torch.norm(x, dim=-1, keepdim=True).clamp_min(self.eps)
        lognorm = torch.log(norm)  # log r

        if d == 2:
            theta = torch.atan2(x[...,1], x[...,0]).unsqueeze(-1)  # angle
            out = torch.cat([lognorm, theta], dim=-1)  # (r, θ)

        elif d == 3:
            theta = torch.atan2(x[...,1], x[...,0]).unsqueeze(-1)  # azimuth
            phi = torch.acos((x[...,2] / norm).clamp(-1+1e-7, 1-1e-7)).unsqueeze(-1)  # polar
            out = torch.cat([lognorm, theta, phi], dim=-1)  # (r, θ, φ)

        else:
            raise NotImplementedError("PolarCoordinates currently supports d=2 or d=3.")

        return out

class MLP(nn.Module):
    def __init__(self,
                input_dim=2,
                index_dim=1,
                hidden_dim=128,
                act=Swish(),
                premodule = None,
                ):
        super().__init__()
        self.input_dim = input_dim
        self.index_dim = index_dim
        self.hidden_dim = hidden_dim
        self.act = act

        self.output_dim = self.input_dim # represent the a vector field so input dim = output dim

        assert premodule is None or premodule in ["PolarCoordinatesWithLogRadius", "NormalizeLogRadius"]
        self.premodule = premodule
        if premodule == "NormalizeLogRadius": 
            self.pre = NormalizeLogRadius()                                    # non - learnable
            self.learnable_network_input_dim = self.input_dim +  1
        elif premodule == "PolarCoordinatesWithLogRadius": 
            self.pre = PolarCoordinatesWithLogRadius() 
            self.learnable_network_input_dim = self.input_dim
        else: # premodule is None
            self.pre = None
            self.learnable_network_input_dim = self.input_dim

        self.main = nn.Sequential(
            nn.Linear(self.learnable_network_input_dim+index_dim, hidden_dim),  # input layer
            act,
            nn.Linear(hidden_dim, hidden_dim),
            act,
            nn.Linear(hidden_dim, hidden_dim),
            act,
            nn.Linear(hidden_dim, self.output_dim),                             # output layer
        )

    def forward(self, input, t):
        # init
        sz = input.size()
        input = input.view(-1, self.input_dim)
        t = t.view(-1, self.index_dim).float()

        # forward
        if self.pre is not None:
            h = self.pre(input)                   # (batch, learnable_network_input_dim)
            h = torch.cat([h, t], dim=1)         # concat index
        else:
            h = torch.cat([input, t], dim=1)     # concat
        output = self.main(h)                    # forward
        return output.view(*sz)

### 2.3. Define evaluate function (compute ELBO)
@torch.no_grad()
def evaluate(gen_sde, x_test):
    gen_sde.eval()
    num_samples_ = x_test.size(0)
    test_elbo = gen_sde.elbo_random_t_slice(x_test)
    gen_sde.train()
    return test_elbo.mean(), test_elbo.std() / num_samples_ ** 0.5
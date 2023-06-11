import numpy as np 
import torch 


class IdentityFilter:

    def __init__(self, param_dims, device):
        return 
    
    def step(self, grad, step, V_hat, Lam_hat, p_idx):
        return step


class MomentumFilter:

    def __init__(self, param_dims, momentum, device):
        self.momentum = momentum
        self.traces = [None for d in param_dims]
        # self.traces = [torch.zeros(*d).to(device) for d in param_dims]
    
    def step(self, grad, step, V_hat, Lam_hat, p_idx):
        if self.traces[p_idx] is None:
            self.traces[p_idx] = step 
        else:
            self.traces[p_idx] = (1 - self.momentum) * step + \
                self.momentum * self.traces[p_idx]
        return self.traces[p_idx]


# class KalmanFilter:

#     def __init__(self, param_dims, alpha, beta):
#         self.dim = dim 
#         self.alpha = alpha 
#         self.beta = beta 

#         self.mu = None 
#         self.Sigma = None 
    
#     def step(grad, step, V_hat, Lam_hat, p_idx):

#         if self.mu is None:
#             self.mu = grad 
#             self.Sigma = (V_hat, Lam_hat)
#             return self.mu, self.Sigma 

#         g = g.reshape(-1,)
#         first_term = V_hat @ (torch.diag(1 / (Lam_hat + self.rho)) @ (V_hat.T @ g))
#         second_term = (g - V_hat@(V_hat.T@g)) / self.rho
#         step = first_term + second_term

#         self.mu = self.Sigma @ ( step +  )


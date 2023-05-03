# References:
# - https://mcneela.github.io/machine_learning/2019/09/03/Writing-Your-Own-Optimizers-In-Pytorch.html
# - https://github.com/amirgholami/adahessian/blob/master/instruction/adahessian.py


import numpy as np
import torch


class BlockSketchySGD:
    
    def __init__(self, model, lr=3e-4, block_rank=3, rho=1e-3):
        self.lr = lr 
        self.model = model
        self.block_rank = block_rank
        self.rho = rho      # Levenberg-Marquardt Regularization
    

    def zero_grad(self):
        for p in self.model.parameters():
            if p.grad is not None:
                p.grad.data.zero_()
    

    def rand_nys_approx(self, Y, Q):    # Y is sketch, Q is test matrix (vecs of HVP)
        p = Y.shape[0] 
        eps = np.spacing(np.linalg.norm(Y))
        # eps = 1e-6
        nu = (p**0.5) * eps
        
        Y_nu = Y + nu*Q 
        C, _ = torch.linalg.cholesky_ex(Q.T @ Y_nu)
        B = torch.linalg.solve_triangular(C, Y_nu.T, upper=False)
        U, Sigma, _ = torch.linalg.svd(B.T, full_matrices=False)
        Lam_hat = torch.clip(Sigma**2 - nu, min=0)
        
        return U, Lam_hat
    
    
    def approx_newton_step(self, V_hat, Lam_hat, g):
        g = g.reshape(-1,)
        first_term = V_hat @ (torch.diag(1 / (Lam_hat + self.rho)) @ (V_hat.T @ g))
        second_term = (g - V_hat@(V_hat.T@g)) / self.rho

        return first_term + second_term
    

    def step(self, loss_tensor):
        # Compute gradients
        gs = torch.autograd.grad(
            loss_tensor, self.model.parameters(), create_graph=True, 
            retain_graph=True
        )

        for g, p in zip(gs, self.model.parameters()):
            # Form sketch
            vs = torch.randn(torch.numel(p), self.block_rank)  # HVP vectors
            vs = torch.linalg.qr(vs, 'reduced').Q    # Reduced / Thin QR

            sketch = []
            for i in range(self.block_rank):
                v = vs[:, i].reshape(*p.shape)
                hvp = torch.autograd.grad(
                    g, p, 
                    grad_outputs=v,
                    only_inputs=True, allow_unused=True, 
                    retain_graph=True
                )
                sketch.append(hvp[0].reshape(-1,))

            with torch.no_grad():
                sketch = torch.stack(sketch).T
                
                # Approx eigendecomposition of Hessian via randomized nystrom method
                V_hat, Lam_hat = self.rand_nys_approx(sketch, vs)

                # Invert using Woodbury formula and get approx Newton step
                step = self.approx_newton_step(V_hat, Lam_hat, g).reshape(*p.shape)
                p.data.add_(-self.lr * step)


# OLD CODE
# For some reason using a function like this does not work; maybe something is 
# being copied that is causing the hvp to become none?
# def _hvp(self, grad, param, vec):

#     hvp = torch.autograd.grad(
#         grad, param, 
#         grad_outputs=vec,
#         only_inputs=True, allow_unused=True, 
#         retain_graph=True
#     )

#     return hvp

# def _make_sketch(self, param, grad):
#     vecs = torch.randn(self.block_rank, torch.numel(param))
#     hvps = []
#     for i in range(self.block_rank):
#         hvps.append(self._hvp(param, grad, vecs[i]))
#     return torch.stack(hvps)

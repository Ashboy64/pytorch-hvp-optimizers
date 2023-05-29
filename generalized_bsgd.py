# References:
# - https://github.com/noahgolmant/pytorch-hessian-eigenthings
# - https://github.com/amirgholami/adahessian/blob/master/instruction/adahessian.py
# - https://mcneela.github.io/machine_learning/2019/09/03/Writing-Your-Own-Optimizers-In-Pytorch.html


import numpy as np
import torch
from filters import * 


class GeneralizedBSGD:
    
    def __init__(self, model, lr=3e-4, block_rank=3, h_recomp_interval=100,
                 filterer=None):
        self.lr = lr 
        self.curr_lrs = [lr for p in model.parameters()]

        if filterer is None:
            self.filterer = IdentityFilter()
        else:
            self.filterer = filterer

        self.model = model
        self.block_rank = block_rank

        self.step_count = 0
        self.h_recomp_interval = h_recomp_interval

        self.cache = [None for p in model.parameters()]
    

    def zero_grad(self):
        for p in self.model.parameters():
            if p.grad is not None:
                p.grad.data.zero_()
    
    @torch.no_grad()
    def rand_nys_approx(self, Y, Q):    # Y is sketch, Q is test matrix (vecs of HVP)
        p = Y.shape[0] 
        # eps = np.spacing(np.linalg.norm(Y))
        eps = 1e-10
        nu = (p**0.5) * eps
        
        Y_nu = Y + nu*Q 

        mid = Q.T @ Y_nu 
        nys_approx = Y_nu @ torch.linalg.solve(mid, Y_nu.T)
        # U, Sigma, V_T = torch.linalg.svd(mid, full_matrices=False)
        # mid_inv = V_T.T @ ((1. / Sigma) * U)
        
        # nys_approx = Y_nu @ mid_inv @ Y_nu.T
        nys_approx_norm = torch.linalg.norm(nys_approx)

        # U, Sigma, V_T = torch.linalg.svd(nys_approx, full_matrices=False)
        # nys_pinv = V_T.T @ ((1. / Sigma) * U)

        # print("Computed nys_pinv")
        
        # return nys_pinv, nys_approx_norm

        Q, R = torch.linalg.qr(nys_approx, mode='reduced')
        Q = Q[:, :self.block_rank]
        R = R[:self.block_rank, :]
        
        
        # P, L, U = torch.linalg.lu(nys_approx)
        # print("Finished PLU")

        return Q, R, nys_approx_norm
    
    def approx_newton_step(self, P, L, U, g):
        print("In newton step")
        g = g.reshape(-1, 1)
        q = P.T @ g 
        print(q)

        q = torch.linalg.solve_triangular(L, q, upper=False)
        print(q)

        print(U)

        q = torch.linalg.solve_triangular(U, q, upper=True)
        print(q)

        return q     
    
    @torch.no_grad()
    def step(self, loss_tensor):
        # Compute gradients
        gs = torch.autograd.grad(
            loss_tensor, self.model.parameters(), create_graph=True, 
            retain_graph=True
        )

        avg_lam_hats = []
        step_mags = []

        for p_idx, (g, p) in enumerate(zip(gs, self.model.parameters())):

            if self.step_count % self.h_recomp_interval == 0:
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
                sketch = torch.stack(sketch).T

                # Approx LU decomposition of Hessian via randomized nystrom method
                self.cache[p_idx] = self.rand_nys_approx(sketch, vs)

            # Invert using Woodbury formula and get approx Newton step
            Q, R, nys_approx_norm = self.cache[p_idx]
            # nys_pinv, nys_approx_norm = self.cache[p_idx]
            # print(f"nys approx norm: {nys_approx_norm}")

            # step = self.approx_newton_step(P, L, U, g).reshape(*p.shape)

            # print(f"step mag before norm: { torch.linalg.norm(step) }")
            # step /= torch.linalg.norm(step)
            # print(f"step mag after norm: { torch.linalg.norm(step) }")

            step = Q.T @ g.reshape(-1,)
            step = R.T @ torch.linalg.solve(R @ R.T, step)
            # step /= torch.linalg.norm(step)
            
            filtered_step = self.filterer.step(g, step, None, None, p_idx).reshape(*p.shape)

            p.data.add_(-self.curr_lrs[p_idx] * filtered_step)
            step_mags.append( torch.linalg.norm(-self.curr_lrs[p_idx] * filtered_step) )
    
        self.step_count += 1

        return {'avg_lrs': self.curr_lrs}

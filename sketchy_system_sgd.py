# References:
# - https://github.com/noahgolmant/pytorch-hessian-eigenthings
# - https://github.com/amirgholami/adahessian/blob/master/instruction/adahessian.py
# - https://mcneela.github.io/machine_learning/2019/09/03/Writing-Your-Own-Optimizers-In-Pytorch.html


import numpy as np
import torch
from filters import * 


class SketchySystemSGD:
    
    def __init__(self, model, lr=3e-4, block_rank=3, cache_interval=100,
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
        self.cache_interval = cache_interval

        self.cache = [None for p in model.parameters()]
    

    def zero_grad(self):
        for p in self.model.parameters():
            if p.grad is not None:
                p.grad.data.zero_()


    @torch.no_grad()
    def step(self, loss_tensor):
        # Compute gradients
        gs = torch.autograd.grad(
            loss_tensor, self.model.parameters(), create_graph=True, 
            retain_graph=True
        )

        step_mags = []
        step_deviations = []

        for p_idx, (g, p) in enumerate(zip(gs, self.model.parameters())):

            if self.step_count % self.cache_interval == 0:
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
                sketch_T = torch.stack(sketch)

                mat = sketch_T @ sketch_T.T 
                mat_rmsn = torch.sum(mat ** 2 / torch.numel(mat))**0.5

                if mat_rmsn > 1e-3:
                    mat_Q, mat_R = torch.linalg.qr(sketch_T.T)
                else:
                    mat_Q, mat_R = None, None

                cache = (vs, sketch_T.T, mat_R)  # (test matrix, sketch, L)
                self.cache[p_idx] = cache

            vs, sketch, mat_R = self.cache[p_idx]

            A = sketch.T
            g = g.reshape(-1, )
            b = vs.T @ g

            # PERTURBATION STEP (take q = g + (grad of constraint violation)
            # q = g + 0.1 * A.T @ (A @ g - b)
            # step_deviations.append( (torch.sum((q - g)**2) / torch.numel(q))**0.5 )
            # q = q.reshape(*p.shape)


            # GRAD PROJECTION STEP (currently this is working the best)
            if mat_R is not None:
                q = torch.linalg.solve_triangular(mat_R.T, (A @ g - b).reshape(-1,1), upper=False)
                q = torch.linalg.solve_triangular(mat_R, q, upper=True).reshape(-1,)
                q = g - A.T @ q
                
                step_deviations.append( (torch.sum((q - g)**2) / torch.numel(q))**0.5 )
            else:
                q = g
                step_deviations.append( 0. )
            q = q.reshape(*p.shape)
            

            # MIN NORM GRADIENT STEP
            # if mat_R is not None:
            #     q = torch.linalg.solve_triangular(mat_R.T, b.reshape(-1,1), upper=False)
            #     q = torch.linalg.solve_triangular(mat_R, q, upper=True).reshape(-1,)
            #     q = A.T @ q

            #     step_deviations.append( (torch.sum((q - g)**2) / torch.numel(q))**0.5 )
            # else:
            #     q = g
            #     step_deviations.append( 0. )
            # q = q.reshape(*p.shape)
            

            q = self.filterer.step(g, q, None, None, p_idx).reshape(*p.shape)

            p.data.add_(-self.curr_lrs[p_idx] * q)
            step_mags.append( torch.linalg.norm(-self.curr_lrs[p_idx] * q) )
        
        self.step_count += 1
        return {'avg_step_mags': step_mags, 'avg_lrs': self.curr_lrs, 'avg_step_deviations': step_deviations}
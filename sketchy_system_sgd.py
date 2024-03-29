# References:
# - https://github.com/noahgolmant/pytorch-hessian-eigenthings
# - https://github.com/amirgholami/adahessian/blob/master/instruction/adahessian.py
# - https://mcneela.github.io/machine_learning/2019/09/03/Writing-Your-Own-Optimizers-In-Pytorch.html


import numpy as np
import torch
from functorch import vmap
from filters import * 


class SketchySystemSGD:
    
    def __init__(self, model, device, variant='grad_proj', lr=3e-4, block_rank=3, cache_interval=100,
                 filterer=None):
        self.lr = lr 
        self.device = device
        self.curr_lrs = [lr for p in model.parameters()]

        self.variant = variant

        if filterer is None:
            self.filterer = IdentityFilter()
        else:
            self.filterer = filterer

        self.model = model
        self.block_rank = block_rank

        self.step_count = 0
        self.cache_interval = cache_interval

        self.cache = [None for p in model.parameters()]

        self.q_hats = [None for p in model.parameters()] 
        self.sim_vars = [None for p in model.parameters()]
    

    def zero_grad(self):
        for p in self.model.parameters():
            if p.grad is not None:
                p.grad.data.zero_()


    @torch.no_grad()
    def step(self, loss_tensor):
        trainable_params = [p for p in self.model.parameters() if p.requires_grad]

        # Compute gradients
        gs = torch.autograd.grad(
            loss_tensor, trainable_params, create_graph=True, 
            retain_graph=True
        )

        step_mags = []
        step_deviations = []
        step_grad_sims = []

        for p_idx, (g, p) in enumerate(zip(gs, self.model.parameters())):

            if not p.requires_grad:
                continue

            if self.step_count % self.cache_interval == 0:
                num_hvps = min(torch.numel(p), self.block_rank)

                # Form sketch
                def get_hvp(v):
                    return torch.autograd.grad(
                            g, p, 
                            grad_outputs=v,
                            only_inputs=True, allow_unused=True, 
                            retain_graph=True
                    )[0].reshape(-1,)
                
                vs = torch.randn(torch.numel(p), num_hvps).to(self.device)  # HVP vectors
                vs = torch.linalg.qr(vs, 'reduced').Q    # Reduced / Thin QR
                sketch_T = vmap(get_hvp)(vs.T.reshape(num_hvps, *p.shape))
                # mat = sketch_T @ sketch_T.T
                
                # mat_rmsn = torch.sum(mat ** 2 / torch.numel(mat))**0.5

                # eigvals = torch.linalg.eigvalsh(mat)
                # print(f"Eigvals = {eigvals}")
                # print(f"Mat rmsn = {mat_rmsn}")

                # if mat_rmsn > 1e-3:
                mat_Q, mat_R = torch.linalg.qr(sketch_T.T)
                # else:
                    # mat_Q, mat_R = None, None

                cache = (vs, sketch_T.T, mat_R)  # (test matrix, sketch, L)
                self.cache[p_idx] = cache

            vs, sketch, mat_R = self.cache[p_idx]

            A = sketch.T
            g = g.reshape(-1, )
            b = -vs.T @ g

            # print(f"HESSIAN CONSTRAINT VIOLATION: { torch.linalg.norm(A @ g - b) }")

            # PERTURBATION STEP (take q = g + (grad of constraint violation)
            if self.variant == 'perturb':
                step_mod = A.T @ (A @ g + b)
                # step_mod /= torch.linalg.norm(step_mod)
                q = -g + step_mod
                
            # GRAD PROJECTION STEP (currently this is working the best)
            elif self.variant == 'grad_proj':
                if mat_R is not None:
                    q = torch.linalg.solve_triangular(mat_R.T, (-A @ g - b).reshape(-1,1), upper=False)
                    q = torch.linalg.solve_triangular(mat_R, q, upper=True).reshape(-1,)
                    q = -g - A.T @ q

                    if torch.any( torch.isnan(q) ):
                        q = -g
                else:
                    q = -g
            
            elif self.variant == 'min_norm':
                if mat_R is not None:
                    q = torch.linalg.solve_triangular(mat_R.T, b.reshape(-1,1), upper=False)
                    q = torch.linalg.solve_triangular(mat_R, q, upper=True).reshape(-1,)
                    q = A.T @ q

                else:
                    q = -g
                    step_deviations.append( 0. )
            
            # Constrain norm to prevent blowup
            q_norm = torch.linalg.norm(q)
            if q_norm > 1.:
                q /= q_norm
            
            # Log deviation from SGD
            step_deviations.append( (torch.sum((q + g)**2) / torch.numel(q))**0.5 )

            
            # Interpolate between gradient and computed update
            # q = (q + g) / 2.
            
            # # Low pass filter on computed updates
            # if self.q_hats[p_idx] is None:
            #     self.q_hats[p_idx] = q 
            # else:
            #     self.q_hats[p_idx] = 0.9*self.q_hats[p_idx] + 0.1*q 
            
            # # Auto lr tuning experiments
            # # Keep track of how closely update directions match to modulate learning rate
            # sim = (torch.dot(self.q_hats[p_idx], q)) / ( torch.linalg.norm(self.q_hats[p_idx]) * torch.linalg.norm(q) )
            # # sim = torch.dot(self.q_hats[p_idx], q)
            # if self.sim_vars[p_idx] is None:
            #     self.sim_vars[p_idx] = sim 
            # else:
            #     self.sim_vars[p_idx] = 0.9*self.sim_vars[p_idx] + 0.1*sim 
            # if self.step_count > 100:
            #     self.curr_lrs[p_idx] = self.lr * self.sim_vars[p_idx]
            
            # # Interpolate between gradient and computed update based on update agreement
            # q = sim*self.q_hats[p_idx] + (1 - sim)*g
            
            # q = self.q_hats[p_idx]

            # step_grad_sims.append( torch.dot(q, g) / ( torch.linalg.norm(g) * torch.linalg.norm(q) ) )

            q = q.reshape(*p.shape)
            q = self.filterer.step(g, q, None, None, p_idx).reshape(*p.shape)

            p.data.add_(self.curr_lrs[p_idx] * q)
            step_mags.append( torch.linalg.norm(-self.curr_lrs[p_idx] * q) )
        
        self.step_count += 1
        return {}
        # return {'avg_step_mags': step_mags, 'avg_lrs': self.curr_lrs, 'avg_step_deviations': step_deviations, 'avg_grad_sims': step_grad_sims}

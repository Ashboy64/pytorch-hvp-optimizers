# References:
# - https://github.com/noahgolmant/pytorch-hessian-eigenthings
# - https://github.com/amirgholami/adahessian/blob/master/instruction/adahessian.py
# - https://mcneela.github.io/machine_learning/2019/09/03/Writing-Your-Own-Optimizers-In-Pytorch.html


import numpy as np
import torch
from filters import * 


class GeneralizedBSGD:
    
    def __init__(self, model, lr=3e-4, block_rank=3, block_size=3, h_recomp_interval=100,
                 filterer=None):
        self.lr = lr 
        self.curr_lrs = [lr for p in model.parameters()]

        if filterer is None:
            self.filterer = IdentityFilter()
        else:
            self.filterer = filterer

        self.model = model
        self.block_rank = block_rank
        self.block_size = block_size

        print(block_rank)
        print(block_size)

        self.step_count = 0
        self.h_recomp_interval = h_recomp_interval

        self.sketches = [None for p in model.parameters()]
        self.cache = [[None for i in range(torch.numel(p) // self.block_size)] \
            for p in model.parameters()]


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

        avg_lam_hats = []
        step_mags = []

        for p_idx, (g, p) in enumerate(zip(gs, self.model.parameters())):
            p_size = torch.numel(p)
            p_step = torch.zeros(p_size)
            num_hvps = min(self.block_rank, p_size)
            
            if self.step_count % self.h_recomp_interval == 0:
                # Form sketch
                vs = torch.randn(torch.numel(p), self.block_rank)  # HVP vectors
                vs = torch.linalg.qr(vs, 'reduced').Q    # Reduced / Thin QR

                sketch = []
                for i in range(num_hvps):
                    v = vs[:, i].reshape(*p.shape)
                    hvp = torch.autograd.grad(
                        g, p, 
                        grad_outputs=v,
                        only_inputs=True, allow_unused=True, 
                        retain_graph=True
                    )
                    sketch.append(hvp[0].reshape(-1,))
                sketch = torch.stack(sketch).T
                
                mid = torch.linalg.pinv(vs.T @ sketch, atol=1e-6)
                self.sketches[p_idx] = (mid, sketch)
            
            g = g.reshape(-1,)

            for block_idx in range(p_size // self.block_size):
                # Extract this block's corresponding portion of sketch
                block_start_idx = block_idx * self.block_size
                block_end_idx = min((block_idx + 1) * self.block_size, p_size)

                # Recompute subsketch pseudoinverse if needed
                if self.step_count % self.h_recomp_interval == 0:
                    mid, sketch = self.sketches[p_idx]
                    subsketch = sketch[block_start_idx : block_end_idx, : ]
                    block_hess_approx = subsketch @ mid @ subsketch.T 
                    inv_block_hess = torch.linalg.pinv(block_hess_approx, atol=1e-6)
                    self.cache[p_idx][block_idx] = inv_block_hess
                
                # Form approx Newton step for this block
                block_step = -self.cache[p_idx][block_idx] @ g[block_start_idx : block_end_idx]
                
                block_step_norm = torch.linalg.norm(block_step)
                if block_step_norm > 1.:
                    block_step /= block_step_norm

                p_step[block_start_idx : block_end_idx] = block_step

            p_step = p_step.reshape(*p.shape)
            filtered_step = self.filterer.step(g, p_step, None, None, p_idx).reshape(*p.shape)
            p.data.add_(self.curr_lrs[p_idx] * filtered_step)
            step_mags.append( torch.linalg.norm(self.curr_lrs[p_idx] * filtered_step) )
            
        self.step_count += 1

        return {'avg_lrs': self.curr_lrs}

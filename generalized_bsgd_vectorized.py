# References:
# - https://github.com/noahgolmant/pytorch-hessian-eigenthings
# - https://github.com/amirgholami/adahessian/blob/master/instruction/adahessian.py
# - https://mcneela.github.io/machine_learning/2019/09/03/Writing-Your-Own-Optimizers-In-Pytorch.html


import numpy as np
import torch
from filters import * 


class GeneralizedBSGDVectorized:
    
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

        self.step_count = 0
        self.h_recomp_interval = h_recomp_interval

        self.sketches = [None for p in model.parameters()]
        self.cache = [[None for i in range(torch.numel(p) // self.block_size)] \
            for p in model.parameters()]

        self.p_idx_to_block_info = {}
        for p_idx, p in enumerate(model.parameters()):
            block_info = []

            p_size = torch.numel(p)
            for i in range(p_size // self.block_size):
                block_start = self.block_size * i 
                block_end = min(self.block_size * (i + 1), p_size)
                block_info.append( list(range(block_start, block_end)) )
            
            block_info = torch.as_tensor(block_info, dtype=torch.long)
            self.p_idx_to_block_info[p_idx] = block_info
        
        self.inv_block_hessians = [None for p in model.parameters()]



    def zero_grad(self):
        for p in self.model.parameters():
            if p.grad is not None:
                p.grad.data.zero_()
    

    def compute_inv_block_hess(self, block_idx_info, p_idx):
        mid, sketch = self.sketches[p_idx]
        subsketch = torch.take_along_dim(sketch, block_idx_info.reshape(-1, 1), dim=0)
        block_hess_approx = subsketch @ mid @ subsketch.T 
        inv_block_hess = torch.linalg.pinv(block_hess_approx, atol=1e-6)
        return inv_block_hess


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

            g_flat = g.reshape(-1,)
            
            if self.step_count % self.h_recomp_interval == 0:
                # Form sketch
                def get_hvp(v):
                    return torch.autograd.grad(
                            g, p, 
                            grad_outputs=v,
                            only_inputs=True, allow_unused=True, 
                            retain_graph=True
                    )[0].reshape(-1,)
                
                vs = torch.randn(torch.numel(p), num_hvps)  # HVP vectors
                vs = torch.linalg.qr(vs, 'reduced').Q    # Reduced / Thin QR
                sketch = torch.vmap(get_hvp)(vs.T.reshape(num_hvps, *p.shape))

                mid = torch.linalg.pinv(sketch @ vs, atol=1e-6)
                self.sketches[p_idx] = (mid, sketch.T)
            
                # Recompute pseudoinverse of block if needed
                self.inv_block_hessians[p_idx] = torch.vmap(lambda x: self.compute_inv_block_hess(x, p_idx=p_idx))(self.p_idx_to_block_info[p_idx])

            # Form approx Newton steps
            def compute_newton_steps(inv_block_hess, block_idx_info):
                block_step = -inv_block_hess @ torch.take_along_dim(g_flat, block_idx_info, dim=0)
                block_step /= torch.clip(torch.linalg.norm(block_step), min=1.)
                return block_step

            p_step = torch.vmap(compute_newton_steps)(self.inv_block_hessians[p_idx], self.p_idx_to_block_info[p_idx])
            p_step = p_step.reshape(*p.shape)
            
            filtered_step = self.filterer.step(g, p_step, None, None, p_idx).reshape(*p.shape)
            p.data.add_(self.curr_lrs[p_idx] * filtered_step)
            step_mags.append( torch.linalg.norm(self.curr_lrs[p_idx] * filtered_step) )
            
        self.step_count += 1

        return {'avg_lrs': self.curr_lrs}

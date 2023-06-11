# References:
# - https://github.com/noahgolmant/pytorch-hessian-eigenthings
# - https://github.com/amirgholami/adahessian/blob/master/instruction/adahessian.py
# - https://mcneela.github.io/machine_learning/2019/09/03/Writing-Your-Own-Optimizers-In-Pytorch.html


import math
import numpy as np
import torch
from functorch import vmap
from filters import * 


class GeneralizedBSGDVectorizedTrunc:
    
    def __init__(self, model, device, lr=3e-4, num_hvps=3, rho=1e-3, block_size=3, 
                 h_recomp_interval=100, filterer=None):
        self.lr = lr 
        self.curr_lrs = [lr for p in model.parameters()]

        if filterer is None:
            self.filterer = IdentityFilter()
        else:
            self.filterer = filterer

        self.model = model
        self.device = device
        self.num_hvps = num_hvps
        self.rho = rho
        self.block_size = block_size

        self.step_count = 0
        self.h_recomp_interval = h_recomp_interval

        self.sketches = [None for p in model.parameters()]

        self.p_idx_to_block_info = {}
        self.p_idx_to_uneven_block = {}

        for p_idx, p in enumerate(model.parameters()):
            block_info = []
            uneven_block_info = None

            p_size = torch.numel(p)
            for i in range(math.ceil(p_size / self.block_size)):
                block_start = self.block_size * i 

                block_end = self.block_size * (i + 1)
                if block_end <= p_size:
                    block_info.append( list(range(block_start, block_end)) )
                else:
                    uneven_block_info = list(range(block_start, p_size))
                    uneven_block_info = torch.as_tensor(uneven_block_info, dtype=torch.long, device=device)
                    # (index info, inv hessian U, inv hessian Lam)
                    self.p_idx_to_uneven_block[p_idx] = (uneven_block_info, None, None)
            
            block_info = torch.as_tensor(block_info, dtype=torch.long, device=device)
            self.p_idx_to_block_info[p_idx] = block_info
        
        self.block_Us = [None for p in model.parameters()]
        self.block_Lams = [None for p in model.parameters()]


    def zero_grad(self):
        for p in self.model.parameters():
            if p.grad is not None:
                p.grad.data.zero_()
    

    def compute_inv_block_hess(self, block_idx_info, C_inv, sketch_T, nu):
        subsketch_T = torch.take_along_dim(sketch_T, block_idx_info.reshape(1, -1), dim=1)

        B = C_inv.T @ subsketch_T
        U, Sigma, _ = torch.linalg.svd(B.T, full_matrices=False)
        Lam_hat = torch.clip(Sigma**2 - nu, min=0)

        return U, Lam_hat

    def compute_newton_step(self, block_idx_info, block_Us, block_Lams, g_flat):
        if len(block_idx_info.shape) == 0:
            return torch.tensor([]).to(block_idx_info.device)
        
        g = -torch.take_along_dim(g_flat, block_idx_info, dim=0)

        first_term = block_Us @ ((1 / (block_Lams + 1e-2)) * (block_Us.T @ g))
        second_term = (g - block_Us@(block_Us.T@g)) / 1e-2

        return first_term + second_term


    @torch.no_grad()
    def step(self, loss_tensor):
        update_hess = (self.step_count % self.h_recomp_interval == 0)

        if update_hess:
            # self.filterer.traces = [None for t in self.filterer.traces]
            loss_tensor.backward(create_graph=True)
        else:
            loss_tensor.backward(create_graph=False)

        avg_lam_hats = []
        step_mags = []

        for p_idx, p in enumerate(self.model.parameters()):
            if not p.requires_grad:
                continue

            g = p.grad
            p_size = torch.numel(p)
            num_hvps = min(self.num_hvps, p_size)

            g_flat = g.reshape(-1,)
            
            if update_hess:
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

                # Shift for stability
                nu = 1e-6 * (sketch_T.shape[1]**0.5)
                sketch_T += nu * vs.T

                eigvals, eigvecs = torch.linalg.eigh(sketch_T @ vs)
                eigvals = eigvals.real
                selected_idxs = eigvals > 1e-6

                eigvals = eigvals[selected_idxs]
                eigvecs = eigvecs[:, selected_idxs]
                C_inv = eigvecs * (eigvals**-0.5).reshape(1, -1)

                # Recompute pseudoinverse of block if needed
                inv_block_hess_fn = lambda x: self.compute_inv_block_hess(block_idx_info=x, C_inv=C_inv, sketch_T=sketch_T, nu=nu)
                self.block_Us[p_idx], self.block_Lams[p_idx] = vmap(inv_block_hess_fn, in_dims=0)(self.p_idx_to_block_info[p_idx])
                
                if p_idx in self.p_idx_to_uneven_block:
                    uneven_idx_info, _, _ = self.p_idx_to_uneven_block[p_idx]
                    uneven_inv_block_hess = \
                        self.compute_inv_block_hess(uneven_idx_info, C_inv=C_inv, sketch_T=sketch_T, nu=nu)
                    self.p_idx_to_uneven_block[p_idx] = (uneven_idx_info, *uneven_inv_block_hess)
                
                del sketch_T

            # Form approx Newton steps
            newton_step_fn = lambda idx_info, block_Us, block_Lams : self.compute_newton_step(idx_info, block_Us, block_Lams, g_flat=g_flat)
            p_step = vmap(newton_step_fn)(self.p_idx_to_block_info[p_idx], self.block_Us[p_idx], self.block_Lams[p_idx]).reshape(-1,)

            if p_idx in self.p_idx_to_uneven_block:
                block_step = self.compute_newton_step(*self.p_idx_to_uneven_block[p_idx], g_flat=g_flat)
                p_step = torch.concat([p_step, block_step])
            
            p_step = p_step.reshape(*p.shape)
            
            # Filter and update
            filtered_step = self.filterer.step(g, p_step, None, None, p_idx).reshape(*p.shape)
            p.data.add_(self.curr_lrs[p_idx] * filtered_step)
            step_mags.append( torch.sqrt(torch.sum((self.curr_lrs[p_idx] * filtered_step)**2) / p_size) )
            
        self.step_count += 1

        return {'avg_lrs': self.curr_lrs} #, 'avg_step_rms': step_mags}

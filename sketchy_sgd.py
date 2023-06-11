import numpy as np
import torch
from functorch import vmap
from filters import * 


class SketchySGD():
    
    def __init__(self, model, device, lr=3e-4, num_hvps=3, rho=1e-3, h_recomp_interval=100,
                 num_auto_lr_iter=10, filterer=None):
        self.lr = lr 
        self.curr_lrs = [lr for p in model.parameters()]
        self.num_auto_lr_iter = num_auto_lr_iter

        if filterer is None:
            self.filterer = IdentityFilter()
        else:
            self.filterer = filterer

        self.model = model
        self.device = device
        self.num_hvps = num_hvps
        self.rho = rho      # Levenberg-Marquardt Regularization

        self.step_count = 0
        self.h_recomp_interval = h_recomp_interval

        self.V_hat = None 
        self.Lam_hat = None
    

    def zero_grad(self):
        for p in self.model.parameters():
            if p.grad is not None:
                p.grad.data.zero_()
    

    def rand_nys_approx(self, Y, Q): 
        """
        Randomized Nystrom Approximation given in SketchySGD Paper.
        
        Variables - Y: sketch of the Hessian
                    Q: orthogonalized test matrix
                    r_k: rank ???
        """
        with torch.no_grad():
            p = Y.shape[0]
            nu = 1e-6 * (p**0.5)
            Y_nu = Y + nu * Q 

            # Project to PSD matrices
            M = Q.T @ Y_nu 
            eigvals, eigvecs = torch.linalg.eigh(M)
            eigvals = eigvals.real
            selected_idxs = eigvals > 1e-6
            eigvals = eigvals[selected_idxs]
            eigvecs = eigvecs[:, selected_idxs]
            
            # Form approx eigendecomp
            C_inv = eigvecs @ torch.diag( eigvals**-0.5 )
            B = (Y_nu @ C_inv).T
            U, Sigma, _ = torch.linalg.svd(B.T, full_matrices=False)
            Lam_hat = torch.clip(Sigma**2 - nu, min=0)
            
            return U, Lam_hat
    
    
    def approx_newton_step(self, V_hat, Lam_hat, g):
        g = g.reshape(-1,)
        first_term = V_hat @ ((1 / (Lam_hat + self.rho)) * (V_hat.T @ g))
        second_term = (g - V_hat@(V_hat.T@g)) / self.rho

        return first_term + second_term
    

    def get_learning_rate(self, p, g, V_hat, Lam_hat, num_iter=10):
        z = torch.randn_like(p)
        y = z / torch.linalg.norm(z)

        lmbda = None 

        for i in range(self.num_auto_lr_iter):
            v = self.approx_newton_step(V_hat, np.sqrt(Lam_hat), y).reshape(*p.shape)
            v_prime = torch.autograd.grad(
                        g, p, 
                        grad_outputs=v,
                        only_inputs=True, allow_unused=True, 
                        retain_graph=True
                    )[0].reshape(-1,)
            new_y = self.approx_newton_step(V_hat, np.sqrt(Lam_hat), v_prime).reshape(-1, 1)
            
            lmbda = y.reshape(1, -1) @ new_y
            y = (new_y / torch.linalg.norm(new_y)).reshape(*p.shape)
        
        return (1 / lmbda).item()


    def _prepare_grad(self, grad_dict):
        """
        Compute gradient w.r.t loss over all parameters and vectorize
        """
        grad_vec = torch.cat([g.contiguous().view(-1) for g in grad_dict])
        return grad_vec
    

    # @torch.no_grad()
    def step(self, loss_tensor):
        trainable_params = [p for p in self.model.parameters() if p.requires_grad]
        grad_dict = torch.autograd.grad(
            loss_tensor, trainable_params, create_graph=True
        )
        grad_vec = self._prepare_grad(grad_dict)
        self.zero_grad()

        if self.step_count % self.h_recomp_interval == 0:
            # Form sketch
            num_hvps = min(self.num_hvps, grad_vec.shape[0])
            vs = torch.randn(grad_vec.shape[0], num_hvps).to(self.device)  # HVP vectors
            vs = torch.linalg.qr(vs, 'reduced').Q    # Reduced / Thin QR

            # Form sketch
            def get_hvp(v):
                hvp_dict = torch.autograd.grad(
                        grad_vec, trainable_params, 
                        grad_outputs=v,
                        only_inputs=True, allow_unused=True, 
                        retain_graph=True
                )
                return torch.cat([g.contiguous().view(-1) for g in hvp_dict])
            sketch_T = vmap(get_hvp)(vs.T.reshape(num_hvps, grad_vec.shape[0]))
            sketch = sketch_T.T
        
            self.V_hat, self.Lam_hat = self.rand_nys_approx(sketch, vs)
        all_steps = self.approx_newton_step(self.V_hat, self.Lam_hat, grad_vec).reshape(-1)
        
        starting_idx = 0
        step_mag = 0.
        for p_idx, p in enumerate(self.model.parameters()):
            if not p.requires_grad:
                continue

            step = -all_steps[starting_idx : starting_idx+torch.numel(p)].reshape(*p.shape)
            p.data.add_(self.lr * step)
            starting_idx += torch.numel(p)
            step_mag += torch.linalg.norm(self.lr * step)**2
        step_mag = step_mag**0.5

        self.step_count += 1

        return {'avg_lrs': self.curr_lrs}

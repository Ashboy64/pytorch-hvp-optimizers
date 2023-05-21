import numpy as np
import torch
from filters import * 


class SketchySGD():
    
    def __init__(self, model, est_hessian=True, lr=3e-4, block_rank=3, rho=1e-3, h_recomp_interval=100,
                 num_auto_lr_iter=10, filterer=None):
        self.est_hessian = est_hessian # if False, recovers the original SketchySGD formulation
        self.lr = lr 
        self.curr_lrs = [lr for p in model.parameters()]
        self.num_auto_lr_iter = num_auto_lr_iter

        if filterer is None:
            self.filterer = IdentityFilter()
        else:
            self.filterer = filterer

        self.model = model
        self.block_rank = block_rank
        self.rho = rho      # Levenberg-Marquardt Regularization

        self.step_count = 0
        self.h_recomp_interval = h_recomp_interval
    

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
            nu = np.sqrt(p) * np.spacing(np.linalg.norm(Y))
            # eps = 1e-6
            Y_nu = Y + nu * Q 
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
    
    @torch.no_grad()
    def step(self, loss_tensor):
        # Compute gradients
        gs = torch.autograd.grad(
            loss_tensor, self.model.parameters(), create_graph=True, 
            retain_graph=True
        )
        
        # Join all the gradients into one long vector
        sketch = []
        stacked_gs = []
        total_elements = 0
        for g, p in zip(gs, self.model_parameters()):
            gs += list(gs)
            total_elements += torch.numel(p)            
            hvp = torch.autograd.grad(
                        g, p, 
                        grad_outputs=v,
                        only_inputs=True, allow_unused=True, 
                        retain_graph=True
                    )
            concatenated_hvp += list(hvp[0].reshape(-1,)) # TODO: make sure this is done correctly..
    
        stacked_gs = torch.FloatTensor(stacked_gs)
        sketch = torch.FloatTensor(concatenated_hvp)
        vs = torch.randn(torch.numel(p), self.block_rank)
        vs = torch.linalg.qr(vs, "reduced").Q
        
        V_hat, Lam_hat = self.rand_nys_approx(sketch, vs)
        
        # TODO: check what Hessian blocks does here..
        
        all_steps = self.approx_newton_step(V_hat, Lam_hat, stacked_gs).reshape(-1)
        starting_idx = 0
        for p in self.model_parameters():
            step = all_steps[starting_idx:starting_idx+torch.numel(p)].reshape(*p.shape)
            p.data.add_(self.lr * filtered_step) #using constant learning rate
            #todo: figure this out too..
            #step_mags.append( torch.linalg.norm(-self.curr_lrs[p_idx] * filtered_step) )
            starting_idx += torch.numel(p)
            
        self.step_count += 1

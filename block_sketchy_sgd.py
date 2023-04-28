# References:
# - https://mcneela.github.io/machine_learning/2019/09/03/Writing-Your-Own-Optimizers-In-Pytorch.html
# - https://github.com/amirgholami/adahessian/blob/master/instruction/adahessian.py


import torch


class BlockSketchySGD:
    
    def __init__(self, model, block_rank=3, lr=3e-4):
        self.lr = lr 
        self.model = model
        self.block_rank = block_rank
    
    def zero_grad(self):
        for p in self.model.parameters():
            if p.grad is not None:
                p.grad.data.zero_()
    
    def rand_nys_approx(self, Y, Q):    # Y is sketch, Q is test matrix (vecs of HVP)
        # Naive implementation
        # return Y @ torch.linalg.solve(Q.T @ Y, Y.T)

        # Sketchy SGD style
        p = Y.shape[0] 
        eps = 1e-6      # TODO: Set according to actual algo
        nu = (p**0.5) * eps
        
        Y_nu = Y + nu*Q 
        print("Computed Y_nu")
        
        # This is where the program hangs
        C, _ = torch.linalg.cholesky_ex(Q.T @ Y_nu)
        print("Computed C")
        
        B = torch.linalg.solve_triangular(C, Y)
        print("Computed B")
        
        V_hat, Sigma, _ = torch.linalg.svd(B, full_matrices=False)
        print("Computed V_hat, Sigma")
        
        Lam_hat = torch.maximum(Sigma@Sigma - nu, 0)
        print("Computed Lam_hat")
        
        return V_hat, Lam_hat


    
    def step(self, loss_tensor, model):
        # Compute gradients
        gs = torch.autograd.grad(
            loss_tensor, model.parameters(), create_graph=True, 
            retain_graph=True
        )

        for g, p in zip(gs, self.model.parameters()):
            # Form sketch
            vs = torch.randn(torch.numel(p), self.block_rank)  # HVP vectors
            vs = torch.linalg.qr(vs, 'reduced').Q.T    # Reduced / Thin QR

            sketch = []
            for i in range(self.block_rank):
                v = vs[i].reshape(*p.shape)
                hvp = torch.autograd.grad(
                    g, p, 
                    grad_outputs=v,
                    only_inputs=True, allow_unused=True, 
                    retain_graph=True
                )
                sketch.append(hvp[0].reshape(-1,))
            sketch = torch.stack(sketch)

            # TODO: This freezes
            V_hat, Lam_hat = self.rand_nys_approx(sketch, vs)
    

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


if __name__ == '__main__':
    hvp_test()
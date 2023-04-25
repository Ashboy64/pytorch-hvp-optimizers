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

    def _hvp(self, param, grad, vec):

        print(grad)
        print(param)
        print(param.grad)

        out = torch.autograd.grad(
            grad.reshape(-1,), param.reshape(-1,), 
            grad_outputs=vec, only_inputs=True, 
            allow_unused=True
        )
        print(out)
        return out 
    
    def _make_sketch(self, param, grad):
        vecs = torch.randn(self.block_rank, torch.numel(param))
        hvps = []
        for i in range(self.block_rank):
            hvps.append(self._hvp(param, grad, vecs[i]))
        return torch.stack(hvps)
    
    def step(self, loss_tensor, model):
        for p in self.model.parameters():
            # g is the gradient of the loss wrt p
            g = torch.autograd.grad(
                loss_tensor, p, create_graph=True
            )

            # Vector with which we are computing HVP
            v = torch.randn(self.block_rank, torch.numel(p))

            hvp = torch.autograd.grad(
                g[0].reshape(-1,), p.reshape(-1,), 
                grad_outputs=torch.ones(torch.numel(p)), only_inputs=True, 
                allow_unused=True
            )

            print(hvp)      # Why is this None???
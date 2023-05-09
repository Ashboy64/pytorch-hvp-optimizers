# Efficient Second Order Neural Network Optimization via Sketching and Block-Diagonal Hessian Approximation

## Usage

`
python main.py dataset=cifar-10 \
               num_epochs=5 \
               model=cnn \
               optimizer=block_sketchy_sgd \
               filter=momentum \
               optimizer.params.lr=3e-4
`
When debugging, add the flag `wandb.mode=disabled` to prevent logging to wandb. CNN and MLP architectures supported
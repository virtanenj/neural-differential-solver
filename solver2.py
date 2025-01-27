import os
import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils as utils
from torchinfo import summary

torch.set_default_dtype(torch.float32)

class LossFunction(nn.Module):
    def __init__(self, equation_operators, boundary_conditions, de_weight=1.0, bc_weight=1.0):
        '''
        equation_operator: Callable
            differential operator F such that F(x, f(x), grad f(x), ..., grad^m f(x)) = 0
            for solution f(x) of the differential equation
        boundary_conditions: List of tuples [(B, xb)], grad^p f(xb) = K(xb)
            B: Callable, boundary condition operator B(xb, grad^p f(xb)) = 0 for some p
            xb: float, boundary condition point
        '''
        super(LossFunction, self).__init__()
        if not isinstance(equation_operators, list):
            equation_operators = [equation_operators]
        self.equation_operators = equation_operators
        self.eq_num = len(equation_operators)
        self.bc_operator = [bc[0] for bc in boundary_conditions]
        self.bc_xb = [bc[1] for bc in boundary_conditions]
        self.bc_num = len(boundary_conditions)
        self.de_weight = de_weight
        self.bc_weight = bc_weight

    def __call__(self, model, x):
        loss = torch.tensor(0.0, device=x.device)
        for eq_i in range(self.eq_num):
            loss += self.equation_operators[eq_i](x, model).square().mean()
        loss *= self.de_weight
        for bc_i in range(self.bc_num):
            bc_term = self.bc_operator[bc_i](self.bc_xb[bc_i], model)
            bc_loss = bc_term.square().mean()
            loss += self.bc_weight * bc_loss
        return loss
    
def save_results(result_dir, model, loss_history):
    # Saves weights, loss history and model summary
    timestamp = time.strftime('%Y%m%d_%H%M%S')
    result_path = os.path.join(result_dir, f'{timestamp}_result')
    os.makedirs(result_path, exist_ok=True)
    np.save(os.path.join(result_path, 'loss_history.npy'), loss_history)
    torch.save(model.state_dict(), os.path.join(result_path, 'model_weights.pth'))
    with open(os.path.join(result_path, 'model_summary.txt'), 'w') as f:
        f.write(str(summary(model, verbose=0)))

def load_model(model, weights_path):
    model.load_state_dict(torch.load(weights_path))
    return model

# TODO ways to avoid seemingly overfitting (even though this should not happen)
# E.g. early stopping, regularization (maybe), batch normalization (maybe), gradient clipping
# TODO learning rate decaying: E.g. use the scheduler from torch.optim.lr_scheduler. In particular ReduceLROnPlateau is recommended. Useful to use together with early stopping.
def train(model, loss_fn, domain, epochs, lr=0.001, batch_size=None, use_scheduler=False, scheduler_factor=0.1, scheduler_patience=10, results_dir=None, print_progress=False,  print_progress_percentage=None, eval_fn=None):

    # Create DataLoader for batching domain
    if batch_size is None:
        batch_size = domain.shape[0] # Use all data for single batch
    dataset = utils.data.TensorDataset(domain)
    dataloader = utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    optimizer = optim.Adam(model.parameters(), lr=lr)

    # Learning rate scheduler TODO tune parameters
    if use_scheduler:
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=scheduler_factor, 
            patience=scheduler_patience, min_lr=1e-6)
    
    loss_history = np.zeros(epochs)
    if eval_fn is not None: 
        eval_history = np.zeros(epochs)

    for epoch_idx in range(epochs):
        epoch_loss = 0.0
        num_batches = 0
        
        for batch in dataloader:
            batch_domain = batch[0] # "Unpack the batch"
            optimizer.zero_grad()
            loss = loss_fn(model, batch_domain)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            num_batches += 1
        
        avg_epoch_loss = epoch_loss / num_batches
        loss_history[epoch_idx] = avg_epoch_loss

        # Update learning rate based on loss
        if use_scheduler: scheduler.step(avg_epoch_loss)

        # optimizer.zero_grad()
        # loss = loss_fn(model, domain)
        # loss.backward()
        # optimizer.step()
        # loss_history[epoch_idx] = loss.item()

        # TODO possible evaluations. E.g.
        # - compute error against true solution
        # - compute output of modelled differential operator F
        if eval_fn is not None:
            pass

        if print_progress:
            if print_progress_percentage is not None:
                if (epoch_idx+1) % int(epochs*print_progress_percentage) == 0:
                    print(f'Epoch {epoch_idx+1}/{epochs} ({(epoch_idx+1)/epochs * 100}%) Loss: {loss.item()}')
            else:
                print(f'Epoch {epoch_idx+1}/{epochs} Loss: {loss.item()}')
    
    if results_dir is not None:
        save_results(results_dir, model, loss_history)

    if eval_fn is not None:
        return loss_history, eval_history
    return loss_history
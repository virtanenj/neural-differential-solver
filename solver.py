import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import grad
from torch.utils.data import DataLoader
from typing import List, Callable


def calcGradients(function, x, order):
    if order == 0:
        return function(x)
    y = function(x)
    grads = grad(outputs=y, inputs=x, grad_outputs=torch.ones_like(y), create_graph=True)[0]
    for _ in range(1, order):
        grads = grad(outputs=grads, inputs=x, grad_outputs=torch.ones_like(grads), create_graph=True)[0]
    return grads


def genDataLoader(domain, discretization_num=None, discretization_delta=None, batch_size=None, batch_shuffle=False):
    assert isinstance(domain, list) and all(isinstance(d, list) and len(d)==2 for d in domain), \
        "`domain` should be a list of lists all with two floats for lower and higher boundry of the dimension."
    if (discretization_num is None) and (discretization_delta is None): 
        raise ValueError("`genDataLoader` expects either `discretization_num: int` or `discretization_delta: int` to be given.")
    elif (discretization_num is not None) and (discretization_delta is not None):
        print(("Warning: both `discretization_num` and `discretization_delta` was given while only either one is used. Defaulting"  
              "to using `discretization_num`."))
    if discretization_num is not None:
        grid_coords = [np.linspace(d[0], d[1], discretization_num) for d in domain]
    elif discretization_delta is not None:
        grid_coords = [np.arange(d[0], d[1], discretization_delta) for d in domain]
    grid = np.meshgrid(*grid_coords)
    x = np.stack([g.flatten() for g in grid], axis=-1)
    x_tensor = torch.from_numpy(x).float().requires_grad_(True)
    if batch_size is None:
        batch_size = len(x)
    dataloader = DataLoader(x_tensor, batch_size=batch_size, shuffle=batch_shuffle)
    return dataloader


def training(model, loss_fn, dataloader, num_epochs, learning_rate, save_model_freq=None, save_model_path=None, print_progress=False):
    if (save_model_path is not None) and (save_model_freq is None):
        raise TypeError("`save_model_path` is not defined while `save_model_freq` is given.")
    
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    loss_history = []

    model.train()  # Make sure to be in train mode
    for epoch_i in range(num_epochs):
        running_loss = 0.0
        for batch_i, batch_x in enumerate(dataloader):
            # Input data in appropriate shape
            batch_x = batch_x.unsqueeze(-1)
            # Backpropagation
            optimizer.zero_grad()
            loss = loss_fn(model, batch_x)
            loss.backward()
            optimizer.step()
            # Keep track of loss
            running_loss += loss.item()
        
        # Potentially save the model at a given intervals
        if save_model_freq is not None:
            if (epoch_i + 1) % save_model_freq == 0:
                torch.save(model.state_dict(), "{}/epoch_{}.pth".format(save_model_path, epoch_i+1))

        loss_history.append(running_loss)
        
        # Print progress
        if print_progress:
            print("Epoch {}/{} Loss: {:.4f}". format(epoch_i+1, num_epochs, running_loss))

    return loss_history


def traininigDomainExtension(model, loss_fn, domains, discretization, num_epochs_per_domain, learning_rate, 
                             save_model_freq=None, save_model_path=None, eval_function=None, eval_freq=1):
    if (save_model_path is not None) and (save_model_freq is None):
        raise TypeError("`save_model_path` is not defined while `save_model_freq` is given.")
    
    num_domains = len(domains)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    loss_history = []
    eval_history = {}  # {epoch: eval function result}
    for domain_i, domain in enumerate(domains):
        dataloader = genDataLoader(domain, discretization)

        model.train()  # Make sure to be in train mode
        for epoch_i in range(num_epochs_per_domain):
            running_loss = 0.0
            for batch_i, batch_x in enumerate(dataloader):
                # Input data in appropriate shape
                batch_x = batch_x.unsqueeze(-1)
                # NB no need to specifically calculate forward push
                # Backpropagation
                optimizer.zero_grad()
                loss = loss_fn(model, batch_x)
                loss.backward()
                optimizer.step()
                # Keep track of loss
                running_loss += loss.item()
            
            # Potentially save the model at a given intervals
            if save_model_freq is not None:
                if (epoch_i + 1) % save_model_freq == 0:
                    torch.save(model.state_dict(), "{}/epoch_{}.pth".format(save_model_path, epoch_i+1))

            loss_history.append(running_loss)
            # Print progress
            print("Domain {}/{}, Epoch {}/{} Loss: {:.4f}". format(domain_i+1, num_domains, epoch_i+1, 
                                                                   num_epochs_per_domain, running_loss))
            
            # Optionally calculate evaluation based on a separate function
            if eval_function is not None:
                if epoch_i % eval_freq == 0:
                    model.eval()
                    # TODO Ability to evaluate current model with a given function (e.g. the correct/analytic solution)
                    # NB remember to then also return the evaluation data alongside with `loss_history`.
                    model.train()
                    pass

    return loss_history


# TODO
def downloadModel():
    pass


# TODO
def loadModel():
    pass


class DNN(nn.Module):
    def __init__(self, input_shape: int, hidden_shapes: List[int], output_shape: int, activations: List[Callable] = None):
        super(DNN, self).__init__()
        depth = len(hidden_shapes)
        if activations is None:
            activations = [nn.Sigmoid() for _ in range(depth)]  # sigmoid as default
        else:
            assert len(activations) == depth
        layers = []
        layers.append(nn.Linear(input_shape, hidden_shapes[0]))
        layers.append(activations[0])
        for layer_i in range(1, depth):
            layers.append(nn.Linear(hidden_shapes[layer_i-1], hidden_shapes[layer_i]))
            layers.append(activations[layer_i])
        layers.append(nn.Linear(hidden_shapes[-1], output_shape))
        self.model = nn.Sequential(*layers)  # unpack list by `*list`

    def forward(self, x: torch.Tensor):
        return self.model(x)


# TODO Explore including convolutional layers to the network
class ConvDNN(nn.Module):
    def __init__(self, input_shape: int, hidden_shapes: List[int], output_shape: int, activations: List[Callable] = None):
        super(DNN, self).__init__()
        depth = len(hidden_shapes)
        if activations is None:
            activations = [nn.Sigmoid() for _ in range(depth)]  # sigmoid as default
        else:
            assert len(activations) == depth
        layers = []
        layers.append(nn.Linear(input_shape, hidden_shapes[0]))
        layers.append(activations[0])
        for layer_i in range(1, depth):
            layers.append(nn.Linear(hidden_shapes[layer_i-1], hidden_shapes[layer_i]))
            layers.append(activations[layer_i])
        layers.append(nn.Linear(hidden_shapes[-1], output_shape))
        self.model = nn.Sequential(*layers)  # unpack list by `*list`

    def forward(self, x: torch.Tensor):
        return self.model(x)


class LossFunction(nn.Module):
    def __init__(self, differential_expressions: List[Callable], bc_boundaries: List[float], 
                 bc_functions: list[Callable], bc_grad_orders: List[int]):
        super(LossFunction, self).__init__()
        self.differential_expressions = differential_expressions
        self.problem_order = len(bc_boundaries)
        self.bc_grad_orders = bc_grad_orders
        self.bc_functions = bc_functions
        bc_boundaries_temp = [float(b) for b in bc_boundaries]
        self.bc_boundaries = torch.tensor(bc_boundaries_temp, requires_grad=True)
        assert self.problem_order == len(self.bc_grad_orders) and self.problem_order == len(self.bc_functions)

    def _calcBoundaryCondition(self, model: nn.Module, x_data: torch.Tensor, include_all_bc: bool):
        bc_terms = []
        for bc_i in range(self.problem_order):
            bc_boundary = self.bc_boundaries[bc_i]
            bc_boundary_inds = (x_data == bc_boundary).nonzero()
            if bc_boundary_inds.numel() > 0:
                x_bc = x_data[bc_boundary_inds[0,0]]
                term1 = calcGradients(model, x_bc, self.bc_grad_orders[bc_i])  # grad_x^p_i model(x_{b_i})
                term2 = self.bc_functions[bc_i](x_bc)  # K_i(x_{p_i})
                if not isinstance(term2, torch.Tensor): 
                    term2 = torch.tensor([term2], requires_grad=True)  # Make sure term2 is tensor which grad is tracked
                # QUESTION If the function `bc_functions[bc_i]` returns a constant, does it matter if the (newly created) 
                # tensor tracks the gradient or not?
                bc_term_i = (term1 - term2)**2
                bc_terms.append(bc_term_i)
        if len(bc_terms) == 0:
            return torch.tensor(0.0, requires_grad=True)
        else:
            return torch.sum(torch.stack(bc_terms))

    def __call__(self, model: nn.Module, x_data: torch.Tensor, include_all_bc: bool = False):
        x_max = x_data.shape[0]
        F_term = self.differential_expressions(model, x_data).square().sum() / x_max
        bc_term = self._calcBoundaryCondition(model, x_data, include_all_bc)
        return F_term + bc_term


if __name__ == "__main__":
    pass
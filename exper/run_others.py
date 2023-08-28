import os
import sys
module_path = os.path.abspath(os.path.join('..'))
if module_path not in sys.path:
    sys.path.append(module_path)

import argparse

import numpy as np

import optuna

import torch
from torch.utils.data import DataLoader
import torch.utils.data as data_utils
from torch.optim import SGD, Adam, Adagrad, Adadelta, RMSprop
from torch_optimizer import Adahessian


import matplotlib.pyplot as plt
# plt.rcParams['figure.facecolor'] = 'white'

from dotenv import load_dotenv
load_dotenv()

from datasets import get_dataset
from loss_fns import get_loss
from utils import solve, restricted_float

torch.set_default_dtype(torch.float64)

# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
device = torch.device('cpu')


datasets_setup = [
    # ["mushrooms", 64, 1.0],
    # ["colon-cancer", 8, 1.0],
    # ["a1a", 64, 1.0],
    # ["covtype.libsvm.binary.scale", 1024, 0.20],
    ["a9a", 2048, 1.0],
    ]


def logistic_reg(w, X, y):
    return torch.mean(torch.log(1 + torch.exp(-y * (X @ w))))

def nllsq(w, X, y):
    return torch.mean( ( y - (1/(1 + torch.exp(-X @ w ))) )**2 )

def save_results(result, dataset, percent, scale, batch_size, epochs, loss_class, optimizer_class, lr, preconditioner, slack_method, lmd, mu, seed):
    results_path = os.getenv("RESULTS_DIR")
    directory = f"{results_path}/{dataset}/percent_{percent}/scale_{scale}/bs_{batch_size}" \
    f"/epochs_{epochs}/{loss_class}/{optimizer_class}/lr_{lr}/precond_{preconditioner}/slack_{slack_method}/lmd_{lmd}/mu_{mu}/seed_{seed}"
    print(directory)
    if not os.path.exists(directory):
        os.makedirs(directory)
        
    torch.save([x[0] for x in result], f"{directory}/loss")
    torch.save([x[1] for x in result], f"{directory}/grad_norm_sq")
    
    if optimizer_class in ("sps", "sps2"):
        torch.save([x[2] for x in result], f"{directory}/slack")


def run(dataset_name, batch_size, percentage, scale, epochs, learning_rate, loss_name, seed):

    torch.random.manual_seed(seed)

    # training 
    STEPS = epochs
    loss_class = get_loss(loss_name)

    scale_k = scale
    # scale_range = [-scale_k, scale_k] # [-value, value]
    train_data, train_target = get_dataset(dataset_name, batch_size, percentage, scale_k)
    train_data = train_data.to(torch.get_default_dtype())
    train_target = train_target.to(torch.get_default_dtype())
    train_load = data_utils.TensorDataset(train_data, train_target)
    train_dataloader = DataLoader(train_load, batch_size=batch_size, shuffle=True)

    lmd = 0.01
    mu = 0.1

    def run_optimizer(optim):
        lr = learning_rate
        if learning_rate == -1:
            study_name = f"{optim.__name__.lower()}-{dataset_name}-{batch_size}-{percentage}-{300}-{loss_name}-{scale_k}"
            storage_name = "sqlite:///optuna_results/{}.db".format(study_name)
            study = optuna.load_study(study_name=study_name, storage=storage_name)
            lr = study.best_params["lr"]
        
        # parameters
        w = torch.zeros(train_data.shape[1], device=device).requires_grad_()
        optimizer = optim([w], lr=lr)
        loss_function = loss_class(w)

        # save loss and grad size to history
        hist = []
        loss = loss_function(train_data.to(device), train_target.to(device))
        g, = torch.autograd.grad(loss, w, create_graph=True)
        print(f"Loss: {loss.item()} | GradNorm^2: {(torch.linalg.norm(g) ** 2 ).item()}")
        hist.append([loss.item(), (torch.linalg.norm(g) ** 2).item()])

        for step in range(STEPS):
            for i, (batch_data, batch_target) in enumerate(train_dataloader):
                batch_data = batch_data.to(device)
                batch_target = batch_target.to(device)
                optimizer.zero_grad()
                loss = loss_function(batch_data, batch_target)
                if isinstance(optimizer, Adahessian): 
                    loss.backward(create_graph=True)
                else:
                    loss.backward()
                optimizer.step()

            loss = loss_function(train_data.to(device), train_target.to(device))
            g, = torch.autograd.grad(loss, w, create_graph=True)
            hist.append([loss.item(), (torch.linalg.norm(g) ** 2).item()])

            if step % 10 == 0 or step == STEPS-1:
                print(f"Epoch [{step}] | Loss: {loss.item()} | GradNorm^2: {(torch.linalg.norm(g) ** 2 ).item()}")

        return hist, lr


    hist_sgd, lr = run_optimizer(SGD)
    save_results(result=hist_sgd, dataset=dataset_name, percent=percentage, scale=scale, batch_size=batch_size, 
                epochs=epochs, loss_class=loss_name, optimizer_class="sgd", lr=round(lr, 5), preconditioner="none", slack_method="none", lmd=lmd, mu=mu, 
                seed=seed)

    hist_adam, lr = run_optimizer(Adam)
    save_results(result=hist_adam, dataset=dataset_name, percent=percentage, scale=scale, batch_size=batch_size, 
                epochs=epochs, loss_class=loss_name, optimizer_class="adam", lr=round(lr, 5), preconditioner="none", slack_method="none", lmd=lmd, mu=mu, 
                seed=seed)

    hist_adahessian, lr = run_optimizer(Adahessian)
    save_results(result=hist_adahessian, dataset=dataset_name, percent=percentage, scale=scale, batch_size=batch_size, 
                epochs=epochs, loss_class=loss_name, optimizer_class="adahessian", lr=round(lr, 5), preconditioner="none", slack_method="none", lmd=lmd, mu=mu, 
                seed=seed)

    hist_adagrad, lr = run_optimizer(Adagrad)
    save_results(result=hist_adagrad, dataset=dataset_name, percent=percentage, scale=scale, batch_size=batch_size, 
                epochs=epochs, loss_class=loss_name, optimizer_class="adagrad", lr=round(lr, 5), preconditioner="none", slack_method="none", lmd=lmd, mu=mu, 
                seed=seed)


    hist_adadelta, lr = run_optimizer(Adadelta)
    save_results(result=hist_adadelta, dataset=dataset_name, percent=percentage, scale=scale, batch_size=batch_size, 
                epochs=epochs, loss_class=loss_name, optimizer_class="adadelta", lr=round(lr, 5), preconditioner="none", slack_method="none", lmd=lmd, mu=mu, 
                seed=seed)

    hist_rmsprop, lr = run_optimizer(RMSprop)
    save_results(result=hist_rmsprop, dataset=dataset_name, percent=percentage, scale=scale, batch_size=batch_size, 
                epochs=epochs, loss_class=loss_name, optimizer_class="rmsprop", lr=round(lr, 5), preconditioner="none", slack_method="none", lmd=lmd, mu=mu, 
                seed=seed)

    


def main(lr, loss_name):
    scales = [0, 3, 5]
    seeds = [0, 1, 2, 3, 4]
    for seed in seeds:
        for scale in scales:
            for dataset_name, batch_size, percentage in datasets_setup:
                print(dataset_name, batch_size, percentage, scale, lr, seed)
                run(dataset_name=dataset_name, batch_size=batch_size, percentage=percentage, scale=scale, epochs=500, learning_rate=lr, loss_name=loss_name, seed=seed)


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Help me!")
    parser.add_argument("--lr", type=float, default=-1.0, help="-1.0 to choose optimized hyperparameter from optuna results.")
    parser.add_argument("--loss", type=str, choices=["logreg", "nllsq"])

    args = parser.parse_args()
    print(f"device: {device}")
    main(lr=args.lr, loss_name=args.loss)
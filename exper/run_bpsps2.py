import os
import sys
module_path = os.path.abspath(os.path.join('..'))
if module_path not in sys.path:
    sys.path.append(module_path)

import argparse

import numpy as np

import torch
from torch.utils.data import DataLoader
import torch.utils.data as data_utils
from torch.optim import SGD, Adam

import matplotlib.pyplot as plt
# plt.rcParams['figure.facecolor'] = 'white'

from dotenv import load_dotenv
load_dotenv()

from datasets import get_dataset
from loss_fns import get_loss
from utils import solve, restricted_float

torch.set_default_dtype(torch.float64)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


datasets_setup = [
    # ["mushrooms", 64, 1.0],
    # ["colon-cancer", 8, 1.0],
    ["covtype.libsvm.binary.scale", 512, 0.1]
    ]


def logistic_reg(w, X, y):
    return torch.mean(torch.log(1 + torch.exp(-y * (X @ w))))

def nllsq(w, X, y):
    return torch.mean( ( y - (1/(1 + torch.exp(-X @ w ))) )**2 )

def rademacher_old(weights):
    return torch.round(torch.rand_like(weights)) * 2 - 1

def diag_estimate_old(weights, grad, iters):
    Ds = []
    for j in range(iters):
        z = rademacher_old(weights)
        with torch.no_grad():
            hvp = torch.autograd.grad(grad, weights, grad_outputs=z, retain_graph=True)[0]
        Ds.append((hvp*z))

    return torch.mean(torch.stack(Ds), 0)

def save_results(result, dataset, percent, scale, batch_size, epochs, loss_class, optimizer_class, lr, preconditioner, slack_method, lmd, mu, seed):
    results_path = os.getenv("RESULTS_DIR")
    directory = f"{results_path}/{dataset}/percent_{percent}/scale_{scale}/bs_{batch_size}" \
    f"/epochs_{epochs}/{loss_class}/{optimizer_class}/lr_{lr}/precond_{preconditioner}/slack_{slack_method}/lmd_{lmd}/mu_{mu}/seed_{seed}"
    print(directory)
    if not os.path.exists(directory):
        os.makedirs(directory)
        
    torch.save([x[0] for x in result], f"{directory}/loss")
    torch.save([x[1] for x in result], f"{directory}/grad_norm_sq")
    
    if optimizer_class in ("sps", "sps2", "bsps2"):
        torch.save([x[2] for x in result], f"{directory}/slack")


def run(dataset_name, batch_size, percentage, scale, epochs):

    torch.random.manual_seed(0)

    # training 
    STEPS = epochs
    loss_name = "logreg"
    loss_class = get_loss(loss_name)

    scale_k = scale
    scale_range = [-scale_k, scale_k] # [-value, value]
    train_data, train_target = get_dataset(dataset_name, batch_size, percentage, scale_range, loss_class.y_range)
    train_data = train_data.to(torch.get_default_dtype())
    train_target = train_target.to(torch.get_default_dtype())
    train_load = data_utils.TensorDataset(train_data, train_target)
    train_dataloader = DataLoader(train_load, batch_size=batch_size, shuffle=True)

    alpha = 1e-4
    beta = 0.999
    lmd = 0.01
    mu = 0.1

    def bpsps2():
        w = torch.zeros(train_data.shape[1], device=device).requires_grad_()

        loss_function = loss_class(w)

        # save loss and grad size to history
        hist_spsb_d = []
        loss = loss_function(train_data.to(device), train_target.to(device))
        g, = torch.autograd.grad(loss, w, create_graph=True)
        print(f"Loss: {loss.item()} | GradNorm^2: {(torch.linalg.norm(g) ** 2 ).item()}")
        hist_spsb_d.append([loss.item(), (torch.linalg.norm(g) ** 2).item(), 0])

        # preconditioninig matrix
        Dk = diag_estimate_old(w, g, 100)
        print("Dk norm: ", torch.linalg.norm(Dk))

        for step in range(STEPS):

            for i, (batch_data, batch_target) in enumerate(train_dataloader):
                batch_data = batch_data.to(device)
                batch_target = batch_target.to(device)

                loss = loss_function(batch_data, batch_target)
                g, = torch.autograd.grad(loss, w, create_graph=True)
            
                vk = diag_estimate_old(w, g, 1)

                # Smoothing and Truncation 
                Dk = beta * Dk + (1 - beta) * vk
                Dk_hat = torch.abs(Dk)
                Dk_hat[Dk_hat < alpha] = alpha

                Dk_hat_inv = 1 / Dk_hat

                sk = torch.randn(batch_data.shape[1], device=device)
                sk = sk / sk.norm()
                yk = Dk_hat_inv * sk
                yk2 = torch.square(Dk_hat_inv) * sk


                # gnorm = (g * Dk_hat_inv).dot(g)

                f_grad = g.clone().detach()
                D_inv = torch.diagflat(Dk_hat_inv.clone().detach())

                D_inv_B = ((yk2.reshape(-1, 1) @ yk.reshape(1, -1)) / (yk.dot(sk)))
                D_inv_B_D_inv = D_inv_B * Dk_hat_inv

                a = torch.dot(f_grad, Dk_hat_inv*f_grad)
                a = a.cpu().detach().numpy()

                b = torch.dot(f_grad, D_inv_B_D_inv@f_grad)
                b = b.cpu().detach().numpy() 

                c_torch = torch.trace(D_inv_B)
                c = c_torch.cpu().detach().numpy()

                AA = 2 * a * c ** 2
                BB = 4 * a * c - 2 * loss.item() * c**2 - b * c
                CC = 2 * a - 4 * c * loss.item() - 2 * b
                DD = - 2 * loss.item()

                
                BB = BB / AA
                CC = CC / AA
                DD = DD / AA

                AA = torch.tensor(1.0)
                
                def lagr(lmd):
                    b = lambda l: D_inv - (l * (D_inv_B_D_inv) / (1 + l * c_torch)) 
                    return lmd * loss -  (1/2)*lmd**2 * torch.dot(f_grad, b(lmd)@f_grad)

                lmds = solve(AA, BB, CC, DD)
                lmds = torch.from_numpy(lmds).to(device)
                                    
                lmd_max = torch.max(lmds)
                lmd_min = torch.maximum(torch.min(lmds), torch.tensor(0))

                lmd_star = lmd_max
                if lagr(lmd_max) < lagr(lmd_min):
                    lmd_star = lmd_min
                
                precond = lmd_star * ( D_inv - (lmd_star * (D_inv_B_D_inv) / (1 + lmd_star * c_torch)) )

                with torch.no_grad():
                    w.sub_(precond  @ f_grad)

            loss = loss_function(train_data.to(device), train_target.to(device))
            g, = torch.autograd.grad(loss, w, create_graph=True)
            hist_spsb_d.append([loss.item(), (torch.linalg.norm(g) ** 2).item(), 0])

            if step % 10 == 0 or step == STEPS-1:
                print(f"Epoch [{step}] | Loss: {loss.item()} | GradNorm^2: {(torch.linalg.norm(g) ** 2 ).item()}")

        return hist_spsb_d

    def bpsps2_l1():
        w = torch.zeros(train_data.shape[1], device=device).requires_grad_()

        loss_function = loss_class(w)

        # slack
        s = torch.tensor(0.0)

        # save loss and grad size to history
        hist_spsb_l1 = []
        loss = loss_function(train_data.to(device), train_target.to(device))
        g, = torch.autograd.grad(loss, w, create_graph=True)
        print(f"Loss: {loss.item()} | GradNorm^2: {(torch.linalg.norm(g) ** 2 ).item()} | s: {s.item()}")
        hist_spsb_l1.append([loss.item(), (torch.linalg.norm(g) ** 2).item(), s.item()])

        # preconditioninig matrix
        Dk = diag_estimate_old(w, g, 100)
        print("Dk norm: ", torch.linalg.norm(Dk))

        for step in range(STEPS):

            for i, (batch_data, batch_target) in enumerate(train_dataloader):
                batch_data = batch_data.to(device)
                batch_target = batch_target.to(device)

                loss = loss_function(batch_data, batch_target)
                g, = torch.autograd.grad(loss, w, create_graph=True)
            
                vk = diag_estimate_old(w, g, 1)

                # Smoothing and Truncation 
                Dk = beta * Dk + (1 - beta) * vk
                Dk_hat = torch.abs(Dk)
                Dk_hat[Dk_hat < alpha] = alpha 

                Dk_hat_inv = 1 / Dk_hat

                sk = torch.randn(batch_data.shape[1], device=device)
                sk = sk / sk.norm()
                yk = Dk_hat_inv * sk
                yk2 = torch.square(Dk_hat_inv) * sk


                s_nil = s - lmd/(2*mu)
                t = loss.item() - s_nil

                f_grad = g.clone().detach()
                D_inv = torch.diagflat(Dk_hat_inv.clone().detach())
                D_inv_B = ((yk2.reshape(-1, 1) @ yk.reshape(1, -1)) / (yk.dot(sk)))
                D_inv_B_D_inv = D_inv_B * Dk_hat_inv

                a = torch.dot(f_grad, Dk_hat_inv * f_grad)
                a = a.cpu().detach().numpy()

                b = torch.dot(f_grad, D_inv_B_D_inv@f_grad)
                b = b.cpu().detach().numpy() 

                c_torch = torch.trace(D_inv_B)
                c = c_torch.cpu().detach().numpy()

                d = 1 + 2 * mu * a

                AA = d * c**2
                BB = 2 * c * d - 2 * mu * t * c**2 - b * c * mu
                CC = d - 4 * mu * t * c - 2 * b * mu
                DD = -2 * mu * t

                BB = BB / AA
                CC = CC / AA
                DD = DD / AA

                AA = torch.tensor(1.0)
                

                def lagr(lmd):
                    b = lambda l: D_inv - (l * (D_inv_B_D_inv) / (1 + l * c_torch)) 
                    return lmd * loss -  (1/2)*lmd**2 * torch.dot(f_grad, b(lmd)@f_grad)

                lmds = solve(AA, BB, CC, DD)
                lmds = torch.from_numpy(lmds).to(device)
                    
                
                lmd_max = torch.max(lmds)
                lmd_star = torch.maximum(lmd_max, torch.tensor(0.0))

                s = torch.maximum(torch.tensor(0.0), s - 1/(2*mu)*(lmd - lmd_star)).item()

                precond = lmd_star * ( D_inv - (lmd_star * (D_inv_B_D_inv) / (1 + lmd_star * c_torch)) )

                with torch.no_grad():
                    w.sub_(precond  @ f_grad)

            loss = loss_function(train_data.to(device), train_target.to(device))
            g, = torch.autograd.grad(loss, w, create_graph=True)
            hist_spsb_l1.append([loss.item(), (torch.linalg.norm(g) ** 2).item(), s])

            if step % 10 == 0 or step == STEPS-1:
                print(f"Epoch [{step}] | Loss: {loss.item()} | GradNorm^2: {(torch.linalg.norm(g) ** 2 ).item()} | s: {s}")

        return hist_spsb_l1

    def bpsps2_l2():
        w = torch.zeros(train_data.shape[1], device=device).requires_grad_()

        loss_function = loss_class(w)

        # slack
        s = torch.tensor(0.0)
        delta = mu + lmd

        # save loss and grad size to history
        hist_spsb_l2 = []
        loss = loss_function(train_data.to(device), train_target.to(device))
        g, = torch.autograd.grad(loss, w, create_graph=True)
        print(f"Loss: {loss.item()} | GradNorm^2: {(torch.linalg.norm(g) ** 2 ).item()} | s: {s.item()}")
        hist_spsb_l2.append([loss.item(), (torch.linalg.norm(g) ** 2).item(), s.item()])

        # preconditioninig matrix
        Dk = diag_estimate_old(w, g, 100)
        print("Dk norm: ", torch.linalg.norm(Dk))

        for step in range(STEPS):

            for i, (batch_data, batch_target) in enumerate(train_dataloader):
                batch_data = batch_data.to(device)
                batch_target = batch_target.to(device)

                loss = loss_function(batch_data, batch_target)
                g, = torch.autograd.grad(loss, w, create_graph=True)
            
                vk = diag_estimate_old(w, g, 1)

                # Smoothing and Truncation 
                Dk = beta * Dk + (1 - beta) * vk
                Dk_hat = torch.abs(Dk)
                Dk_hat[Dk_hat < alpha] = alpha

                Dk_hat_inv = 1 / Dk_hat

                sk = torch.randn(batch_data.shape[1], device=device)
                sk = sk / sk.norm()
                yk = Dk_hat * sk
                yk2 = torch.square(Dk_hat_inv) * sk


                s_nil = s - lmd/(2*mu)
                t = loss.item() - s_nil

                f_grad = g.clone().detach()
                D_inv = torch.diagflat(Dk_hat_inv.clone().detach())
                D_inv_B = ((yk2.reshape(-1, 1) @ yk.reshape(1, -1)) / (yk.dot(sk)))
                D_inv_B_D_inv = D_inv_B * Dk_hat_inv


                a = torch.dot(f_grad, Dk_hat_inv*f_grad)
                a = a.cpu().detach().numpy()

                b = torch.dot(f_grad, D_inv_B_D_inv@f_grad)
                b = b.cpu().detach().numpy() 

                c_torch = torch.trace(D_inv_B)
                c = c_torch.cpu().detach().numpy()

                d = a + 1/(2 * delta) 

                AA = 2 * d * c**2
                BB = 4 * c * d - 2 * t * c**2 - b * c
                CC = 2 * d - 4 * t * c - 2 * b
                DD = -2 * t

                BB = BB / AA
                CC = CC / AA
                DD = DD / AA

                AA = torch.tensor(1.0)

                lmds = solve(AA, BB, CC, DD)
                lmds = torch.from_numpy(lmds).to(device)
                                    
                lmd_max = torch.max(lmds)
                lmd_star = torch.maximum(lmd_max, torch.tensor(0.0))

                s = (1/delta * (mu * s + lmd_star/2)).item()

                precond = lmd_star * ( D_inv - (lmd_star * (D_inv_B_D_inv) / (1 + lmd_star * c_torch)) )

                with torch.no_grad():
                    w.sub_(precond  @ f_grad)

            loss = loss_function(train_data.to(device), train_target.to(device))
            g, = torch.autograd.grad(loss, w, create_graph=True)
            hist_spsb_l2.append([loss.item(), (torch.linalg.norm(g) ** 2).item(), s])

            if step % 10 == 0 or step == STEPS-1:
                print(f"Epoch [{step}] | Loss: {loss.item()} | GradNorm^2: {(torch.linalg.norm(g) ** 2 ).item()} | s: {s}")

        return hist_spsb_l2


    hist_bpsps2 = bpsps2()
    save_results(result=hist_bpsps2, dataset=dataset_name, percent=percentage, scale=scale, batch_size=batch_size, 
                epochs=epochs, loss_class=loss_name, optimizer_class="bsps2", lr=0.1, preconditioner="hutch", slack_method="none", lmd=lmd, mu=mu, 
                seed=0)

    hist_bpsps2_l1 = bpsps2_l1()
    save_results(result=hist_bpsps2_l1, dataset=dataset_name, percent=percentage, scale=scale, batch_size=batch_size, 
                epochs=epochs, loss_class=loss_name, optimizer_class="bsps2", lr=0.1, preconditioner="hutch", slack_method="L1", lmd=lmd, mu=mu, 
                seed=0)

    hist_bpsps2_l2 = bpsps2_l2()
    save_results(result=hist_bpsps2_l2, dataset=dataset_name, percent=percentage, scale=scale, batch_size=batch_size, 
                epochs=epochs, loss_class=loss_name, optimizer_class="bsps2", lr=0.1, preconditioner="hutch", slack_method="L2", lmd=lmd, mu=mu, 
                seed=0)



def main():
    scales = [0, 3]
    for scale in scales:
        for dataset_name, batch_size, percentage in datasets_setup:
            print(dataset_name, batch_size, percentage, scale)
            run(dataset_name=dataset_name, batch_size=batch_size, percentage=percentage, scale=scale, epochs=200)


if __name__ == "__main__":
    main()




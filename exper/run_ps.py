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
from optimizers import PSPS

from datasets import get_dataset    
from loss_fns import get_loss
from utils import solve, restricted_float

from dotenv import load_dotenv
load_dotenv()

torch.set_default_dtype(torch.float64)

device = torch.device('cpu')


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

def citardouq_solve(a, b, c):
    det = b * b - 4 * a * c
    if det < 1e-40:
        x1 = 0.0
        x2 = 0.0
    else:
        x1 = (2 * c) / (-b - np.sqrt(det))
        x2 = (2 * c) / (-b + np.sqrt(det))
    return np.asarray([x1, x2])


# parameters for D
alpha = 1e-4
beta=0.999
# parameters for slack
lmd = 0.01
mu = 0.1


optimizers_dict = {
    "psps": PSPS
}


def train(seed, loss, train_data, train_target, batch_size, EPOCHS, optimizer_class, **optimizer_kwargs):
    
    torch.random.manual_seed(seed)

    params = torch.zeros(train_data.shape[1], device=device).requires_grad_()

    train_load = data_utils.TensorDataset(train_data, train_target)
    train_dataloader = DataLoader(train_load, batch_size=batch_size, shuffle=True)

    criterion = loss(params)
    optimizer = optimizer_class([params], **optimizer_kwargs)

    train_loss = criterion(train_data.to(device), train_target.to(device))
    g, = torch.autograd.grad(train_loss, params)
    grad_norm_sq = torch.linalg.norm(g).item() ** 2
    slack = 0
    step_size = 0


    hist = [[train_loss.item(), grad_norm_sq, slack, step_size]]
   
    for epoch in range(EPOCHS):
        for i, (batch_data, batch_target) in enumerate(train_dataloader):  
            batch_data = batch_data.to(device)
            batch_target = batch_target.to(device)

            optimizer.zero_grad()
            
            def closure():
                return criterion(batch_data, batch_target)
            
            loss = closure()    
            optimizer.step(closure) 
            slack = optimizer.replay_buffer[-1]["slack"]
            step_size = optimizer.replay_buffer[-1]["step_size"]

        train_loss = criterion(train_data.to(device), train_target.to(device))
        g, = torch.autograd.grad(train_loss, params)
        grad_norm_sq = torch.linalg.norm(g).item() ** 2

        hist.append([train_loss.item(), grad_norm_sq, slack, step_size])

        if epoch % 100 == 0:
            print(f"[{epoch}] / [{EPOCHS}] | Loss: {train_loss.item()} | GradNorm^2: {grad_norm_sq} | s:{slack} | step_size: {step_size}")

    return hist



def train_psps2(seed, train_data, train_target, train_dataloader, loss_class, EPOCHS, update_precond, constr_precond, slack_method):
    
    torch.random.manual_seed(seed)

    # parameters
    alpha = 1e-4
    beta=0.999
    w = torch.zeros(train_data.shape[1], device=device).requires_grad_()

    loss_function = loss_class(w)

    # slack
    s = torch.tensor(0.0)
    delta = mu + lmd

    # save loss and grad size to history
    hist = []
    loss = loss_function(train_data.to(device), train_target.to(device))
    g, = torch.autograd.grad(loss, w, create_graph=True)
    f_grad = g.clone().detach()
    print(f"Loss: {loss.item()} | GradNorm^2: {(torch.linalg.norm(g) ** 2 ).item()} | s: {s.item()}")

    lmd_star = torch.tensor(0.0)
    hist.append([loss.item(), (torch.linalg.norm(g) ** 2).item(), s.item(), lmd_star.item()])

    assert constr_precond != "none", "Constraint preconditioner cannot be Identity"

    # preconditioninig matrix
    Dk = diag_estimate_old(w, g, 100)
    MAX_ITER = 100


    for epoch in range(EPOCHS):

        for i, (batch_data, batch_target) in enumerate(train_dataloader):

            batch_data = batch_data.to(device)
            batch_target = batch_target.to(device)

            loss = loss_function(batch_data, batch_target)
            g, = torch.autograd.grad(loss, w, create_graph=True)
            f_grad = g.clone().detach()

            s_nil = s - lmd/(2*mu)
            t = loss.item() - s_nil

            if update_precond == "hutch":
                vk = diag_estimate_old(w, g, 1)
                # Smoothing and Truncation 
                Dk = beta * Dk + (1 - beta) * vk
                Dk_hat = torch.abs(Dk)
                Dk_hat[Dk_hat < alpha] = alpha
                Dk_hat_inv = 1 / Dk_hat 

                if constr_precond == "hutch":
                    gnorm_sq = (f_grad * Dk_hat_inv).dot(f_grad)
                    a = torch.dot(f_grad, Dk_hat_inv*f_grad).cpu().detach().numpy()

                    if slack_method == "none":
                        det = 1 - (2 * loss.item() / gnorm_sq )
                        if det <= 1e-40:
                            continue
                        else:
                            t = torch.sqrt(det)/det
                            root1 = -1 + t
                            root2 = -1 - t
                            lmd_star = torch.maximum(root1, root2)
                    elif slack_method == "L1":
                        AA = 1
                        BB = 2 + mu * a - 2 * mu * t
                        CC = 1 + 2 * mu * a - 4 * mu * t
                        DD = -2 * mu * t

                        roots = solve(AA, BB, CC, DD)
                        try:
                            roots = torch.from_numpy(roots)      
                        except TypeError as err:
                            print(roots, err)
                            continue
                        lmd_star = torch.relu(torch.max(roots))

                        s = torch.relu(s - ((lmd - lmd_star)/2 * mu)).item()
                    elif slack_method == "L2":
                        AA = 1
                        BB = 2 + delta * a - 2 * delta * t
                        CC = 1 + 2 * delta * a - 4 * delta * t
                        DD = -2 * delta * t

                        roots = solve(AA, BB, CC, DD)
                        try:
                            roots = torch.from_numpy(roots)      
                        except TypeError as err:
                            print(roots, err)
                            continue      
                        lmd_star = torch.relu(torch.max(roots))

                        s = (1/delta) * (mu * s + (lmd_star/2)).item()

                    precond = lmd_star/(1 + lmd_star) * Dk_hat_inv    
                    with torch.no_grad():
                        w.sub_(precond  * f_grad)

                        

                elif constr_precond == "sr1":
                    sk = f_grad.clone()
                    yk = torch.autograd.grad(g, w, grad_outputs=sk, retain_graph=True)[0]
                    gnorm = (g * Dk_hat_inv).dot(g)

                    if gnorm < 1e-25:
                        continue

                    D_inv = torch.diagflat(Dk_hat_inv.clone().detach())

                    D_inv_B = (((Dk_hat_inv * yk).reshape(-1, 1) @ yk.reshape(1, -1)) / (yk.dot(sk)))
                    D_inv_B_D_inv = D_inv_B * Dk_hat_inv

                    a_torch = torch.dot(f_grad, Dk_hat_inv*f_grad)
                    a = a_torch.cpu().detach().numpy()

                    b_torch = torch.dot(f_grad, D_inv_B_D_inv@f_grad)
                    b = b_torch.cpu().detach().numpy() 

                    c_torch = torch.trace(D_inv_B)
                    c = c_torch.cpu().detach().numpy()

                    if slack_method == "L1":
                        d = a + (1 / 2 * mu)
                    elif slack_method == "L2":
                        d = a + 1/(2 * delta)

                    if slack_method == "none":

                        AA = 2 * c**2 * ( torch.dot(f_grad, (D_inv - (1/c) * D_inv_B_D_inv ) @ f_grad) )
                        BB = c_torch * ( torch.dot(f_grad, (4 * D_inv - (3/c) * D_inv_B_D_inv) @ f_grad) )
                        CC = 2 * a - 4 * c * loss.item()
                        DD = - 2 * loss.item()

                        BB = BB / AA
                        CC = CC / AA
                        DD = DD / AA
                        AA = 1.0
                        
                        def lagr(lmd):
                            b = lambda lmd: D_inv - (lmd * (D_inv_B_D_inv) / (1 + lmd * c_torch)) 
                            return lmd * loss -  (1/2)*lmd**2 * torch.dot(f_grad, b(lmd)@f_grad)

                        lmd_star_old = lmd_star
                        lmds = solve(AA, BB, CC, DD)
                        try:
                            lmds = torch.from_numpy(lmds).to(device)
                        except TypeError as err:
                            print(lmds, err)
                            continue
                        lmd_max = torch.max(lmds)
                        lmd_min = torch.maximum(torch.min(lmds), torch.tensor(0))
                        lmd_star = lmd_max
                        if lagr(lmd_max) < lagr(lmd_min):
                            lmd_star = lmd_min

                        if lmd_star > 1e8:
                            print(f"lmd_star is: {lmd_star}")
                            lmd_star = lmd_star_old
                        
                        precond = lmd_star * ( D_inv - (lmd_star * (D_inv_B_D_inv) / (1 + lmd_star * c_torch)) )

                        with torch.no_grad():
                            w.sub_(precond @ f_grad)

                    else:
                        # AA = 2 * d * c**2 - 2 * b * c
                        # BB = 4 * d * c - 3 * b - 2 * t * c**2
                        AA = 2 * c_torch * ((c/2*mu) + torch.dot(f_grad, (c_torch * D_inv - D_inv_B_D_inv) @ f_grad))
                        BB = ( 2 * c_torch * (1 - t * c * mu) / mu ) + torch.dot(f_grad, (4 * c_torch * D_inv - 3 * D_inv_B_D_inv) @ f_grad) 
                        CC = 2 * d - 4 * t * c
                        DD = -2 * t

                        BB = BB / AA
                        CC = CC / AA
                        DD = DD / AA
                        AA = torch.tensor(1.0)

                        lmds = solve(AA, BB, CC, DD)
                        try:
                            lmds = torch.from_numpy(lmds).to(device)
                        except TypeError as err:
                            print(lmds, err)
                            continue
                        lmd_max = torch.max(lmds)
                        lmd_min = torch.maximum(torch.min(lmds), torch.tensor(0))
                        lmd_star = lmd_max

                        if lmd_star > 1e6:
                            print(f"lmd_star is: {lmd_star}")
                            pass
                        else:
                            precond = lmd_star * ( D_inv - (lmd_star * (D_inv_B_D_inv) / (1 + lmd_star * c_torch)) )
                            if slack_method == "L1":
                                s = torch.maximum(torch.tensor(0.0), s - 1/(2*mu)*(lmd - lmd_star)).item()
                            elif slack_method == "L2":
                                s = (1/delta * (mu * s + lmd_star/2)).item()

                        with torch.no_grad():
                            w.sub_(precond  @ f_grad)


            if update_precond == "sr1":

                if constr_precond == "hutch":

                    vk = diag_estimate_old(w, g, 1)
                    # Smoothing and Truncation 
                    Dk = beta * Dk + (1 - beta) * vk
                    Dk_hat = torch.abs(Dk)
                    Dk_hat[Dk_hat < alpha] = alpha
                    Dk_hat_inv = 1 / Dk_hat

                    sk = f_grad.clone()
                    yk = torch.autograd.grad(g, w, grad_outputs=sk, retain_graph=True)[0]
                    gnorm_sq = (g * Dk_hat_inv).dot(g)

                    D_inv = torch.diagflat(Dk_hat_inv.clone().detach())

                    D_inv_B = (((Dk_hat_inv * yk).reshape(-1, 1) @ yk.reshape(1, -1)) / (yk.dot(sk)))
                    D_inv_B_D_inv = D_inv_B * Dk_hat_inv

                    a_torch = torch.dot(f_grad, Dk_hat_inv*f_grad)
                    a = a_torch.cpu().detach().numpy()

                    b_torch = torch.dot(f_grad, D_inv_B_D_inv@f_grad)
                    b = b_torch.cpu().detach().numpy() 

                    c_torch = torch.trace(D_inv_B)
                    c = c_torch.cpu().detach().numpy()
                    
                    d = torch.dot(f_grad, (D_inv_B @ D_inv_B_D_inv) @ f_grad)
                    
                    det = d / (a - 2 * loss.item())
                    if det < 0.0:
                        print(det)
                        continue

                    root_1 = -c + torch.sqrt(det)
                    root_2 = -c - torch.sqrt(det)
                    lmd_star = torch.maximum(torch.tensor(0.0), torch.maximum(root_1, root_2))
                    
                    if lmd_star > 1e8:
                        print(f"lmd_star is: {lmd_star}")
                        continue

                    precond = D_inv - (D_inv_B_D_inv / (lmd_star + c_torch))

                    with torch.no_grad():
                        w.sub_(precond @ f_grad)

                # CG is here
                elif constr_precond == "sr1":
                    hgp = torch.zeros_like(w) # s = H_inv * grad
                    r = f_grad.clone()
                    p = r.detach().clone()

                    for cg_step in range(MAX_ITER):
                        hvp = torch.autograd.grad(g, w, grad_outputs=p, retain_graph=True)[0]
                        alpha_k = torch.dot(r, r) / torch.dot(p, hvp)
                        hgp = hgp + alpha_k * p
                        r_prev = r.clone()
                        r = r - alpha_k * hvp
                        if torch.norm(r) < 1e-10:
                            Ax = torch.autograd.grad(g, w, grad_outputs=hgp, retain_graph=True)[0]    
                            diff = torch.norm(Ax - f_grad)
                            print(f"Took {cg_step} to reach diff={diff}")
                            break

                        beta_k = torch.dot(r, r) / torch.dot(r_prev, r_prev)
                        p = r + beta_k * p
                    
                    gnorm_sq = torch.dot(f_grad, hgp)
                    if slack_method == "none":
                        det = (-gnorm_sq) / (2 * loss.item() - gnorm_sq)
                        if det < 0.0:
                            print(f"det: {det}")
                            continue
                        root_1 = -1 + torch.sqrt(det)
                        root_2 = -1 - torch.sqrt(det)
                        lmd_star = torch.maximum(torch.tensor(0.0), torch.maximum(root_1, root_2))
                        step = lmd_star / (1 + lmd_star) * hgp
                        with torch.no_grad():
                            w.sub_(step)
                    else:
                        if slack_method == "L1":
                            AA = 1
                            BB = 2 + mu * gnorm_sq - 2 * mu * t
                            CC = 1 + 2 * mu * gnorm_sq - 4 * mu * t
                            DD = -2 * mu * t
                            
                            roots = solve(AA, BB, CC, DD)
                            try:
                                roots = torch.from_numpy(roots)      
                            except TypeError as err:
                                print(roots, err)
                                continue
                            lmd_star = torch.relu(torch.max(roots))

                        elif slack_method == "L2":
                            AA = 1
                            BB = 2 + delta * gnorm_sq - 2 * delta * t
                            CC = 1 + 2 * delta * gnorm_sq - 4 * delta * t
                            DD = -2 * delta * t

                            roots = solve(AA, BB, CC, DD)
                            try:
                                roots = torch.from_numpy(roots)      
                            except TypeError as err:
                                print(roots, err)    
                                continue
                            lmd_star = torch.relu(torch.max(roots))

                        if lmd_star > 1e2:
                            print(f"lmd_star is: {lmd_star}")
                            continue
                    
                        if slack_method == "L1":
                            s = torch.relu(s - ((lmd - lmd_star)/2 * mu)).item()
                        elif slack_method == "L2":
                            s = (1/delta) * (mu * s + (lmd_star/2)).item()
                        precond = lmd_star/(1 + lmd_star) * hgp
                            
                        with torch.no_grad():
                            w.sub_(precond)


        loss = loss_function(train_data.to(device), train_target.to(device))
        g, = torch.autograd.grad(loss, w, create_graph=True)
        if epoch % 100 == 0:
            print(f"[{epoch}/{EPOCHS}] | Loss: {loss.item()} | GradNorm^2: {(torch.linalg.norm(g) ** 2 ).item()} | s: {s}")
        hist.append([loss.item(), (torch.linalg.norm(g) ** 2).item(), s, lmd_star.item()])

                

    return hist





def train_sp2plus(seed, train_data, train_target, train_dataloader, loss_class, EPOCHS, slack_method):
    torch.random.manual_seed(seed)

    w = torch.zeros(train_data.shape[1], device=device).requires_grad_()
    s = torch.tensor(0.0)


    loss_function = loss_class(w)
    # save loss and grad size to history
    hist = []
    loss = loss_function(train_data.to(device), train_target.to(device))
    g, = torch.autograd.grad(loss, w, create_graph=True)
    print(f"Loss: {loss.item()} | GradNorm^2: {(torch.linalg.norm(g) ** 2 ).item()} | s: {s.item()}")
    hist.append([loss.item(), (torch.linalg.norm(g) ** 2).item(), s.item()])
        
    for epoch in range(EPOCHS):

        for i, (batch_data, batch_target) in enumerate(train_dataloader):

            batch_data = batch_data.to(device)
            batch_target = batch_target.to(device)

            loss = loss_function(batch_data, batch_target)
            g, = torch.autograd.grad(loss, w, create_graph=True)
            f_grad = g.clone().detach()
            gnorm_sq = torch.norm(f_grad)**2

            loss_closure = lambda w: loss_function(batch_data, batch_target)
            hgp = torch.autograd.functional.hvp(loss_closure, w, g, create_graph=True)[1]

            if slack_method == "none":
                with torch.no_grad():
                    sps_step = loss.item() / gnorm_sq
                    w.sub_(sps_step * f_grad)
                    gdiffHgp = torch.sub(f_grad, hgp, alpha=sps_step)
                    if torch.norm(gdiffHgp)**2 > 1e-10:
                            w.sub_(0.5 * (sps_step**2) * gdiffHgp * torch.dot(f_grad, gdiffHgp)/ (torch.norm(gdiffHgp)**2))


            elif slack_method == "L1":
                Gamma3 = torch.relu(loss.item() - (s - (lmd / (2 * (1 - lmd) ))))
                Gamma3 /= 1 + gnorm_sq

                Gamma4 = torch.minimum(Gamma3, loss.item() / gnorm_sq)

                Lambda1 = loss.item() - Gamma4 * gnorm_sq + 0.5 * Gamma4 * Gamma4 * torch.dot(hgp, f_grad)

                Gamma5 = torch.relu(Lambda1 - (s - lmd / (2 * (1 - lmd))))
                t_norm_sq = 1 + torch.norm( f_grad - Gamma4 * hgp  ) ** 2
                Gamma5 /= t_norm_sq

                Gamma6 = torch.minimum(Gamma5, Lambda1 / t_norm_sq )
                sps_step = (Gamma4 + Gamma6)
                ps2_step = Gamma6 * Gamma4
                with torch.no_grad():                
                    w.sub_(sps_step * f_grad)
                    if torch.norm(ps2_step)**2 > 1e-10:
                            w.add_(ps2_step * hgp)

                s = torch.relu( torch.relu(s - (lmd / (2 * (1 - lmd))) + Gamma3) - (lmd / (2 * (1 - lmd))) + Gamma5)

            elif slack_method == "L2":
                Gamma1 = torch.relu( loss.item() - (1 - lmd) * s ) / (1 - lmd + gnorm_sq)
                t1 = loss.item() - Gamma1 * gnorm_sq - (1 - lmd)**2 * (s + Gamma1) + 0.5 * Gamma1 * Gamma1 * torch.dot(hgp, f_grad)
                t2 = 1 - lmd + torch.norm(f_grad - Gamma1 * hgp)**2
                Gamma2 = torch.relu(t1 / t2)
                sps_step = Gamma1 + Gamma2
                ps2_step = Gamma2 * Gamma1
                
                with torch.no_grad():                
                    w.sub_(sps_step * f_grad)
                    if torch.norm(ps2_step)**2 > 1e-10:
                        w.add_(ps2_step * hgp)


                s = (1 - lmd) * ((1 - lmd) * (s + Gamma1) + Gamma2)


        loss = loss_function(train_data.to(device), train_target.to(device))
        g, = torch.autograd.grad(loss, w, create_graph=True)
        if epoch % 100 == 0:
            print(f"[{epoch} / {EPOCHS}] | Loss: {loss.item()} | GradNorm^2: {(torch.linalg.norm(g) ** 2 ).item()} | s: {s.item()}")

        hist.append([loss.item(), (torch.linalg.norm(g) ** 2).item(), s.item()])

    return hist


def main(dataset_name, batch_size, percentage, scale, loss_name, optimizer_class, EPOCHS, lr, preconditioner, 
        slack_method, lmd, mu, seed, save):
    
 
    torch.random.manual_seed(seed)

    if dataset_name == "synthetic":
        np.random.seed(seed)
        n = 1000
        d = 100
        dataset_name = f"synthetic-{n}x{d}"
        A = np.random.randn(n,d)

        U, S, VH = np.linalg.svd(A)
        S *= 0.0
        S = np.asarray([1/((x+1)**2) for x in range(S.shape[0])])
        A = np.dot(U[:, :S.shape[0]] * S, VH)
        dataset_name += "-modified"

        print(dataset_name)

        xopt = np.random.randn(d)
        b = A @ xopt 
        train_data = torch.Tensor(A)
        train_target = torch.Tensor(b)
        xopt = torch.Tensor(xopt)

        if scale != 0:
            r1 = -scale
            r2 = scale
            scaling_vec = (r1 - r2) * torch.rand(train_data.shape[1]) + r2
            scaling_vec = torch.pow(torch.e, scaling_vec)
            train_data = scaling_vec * train_data

        # EPOCHS = 1000
        train_load = torch.utils.data.TensorDataset(train_data, train_target)
        # batch_size = 128
        train_dataloader = torch.utils.data.DataLoader(train_load, batch_size=batch_size, shuffle=True)

    else:
    
        # Setup
        uninterpolatible = False

        # training 
        # EPOCHS = 500
        train_data, train_target = get_dataset(dataset_name, batch_size, percentage, scale)

        X = train_data
        Y = train_target
        if uninterpolatible:
            X = torch.cat((train_data, train_data, train_data))
            Y = torch.cat((train_target, train_target, -train_target))
            
        train_data = X.to(torch.get_default_dtype())
        train_target = Y.to(torch.get_default_dtype())
        train_load = data_utils.TensorDataset(train_data, train_target)
        train_dataloader = DataLoader(train_load, batch_size=batch_size, shuffle=True)

    loss_class = get_loss(loss_name)
    
    if optimizer_class == "psps":
        result = train(
            seed,
            loss_class,
            train_data, 
            train_target, 
            batch_size,
            EPOCHS,
            PSPS,
            preconditioner=preconditioner,
            slack_method=slack_method,
            lmd=lmd,
            mu=mu
        ) 
    elif optimizer_class.split("_")[0] == "psps2":
        precond = optimizer_class.split("_")[1]
        if precond == "DD":
            update_precond, constr_precond = "hutch", "hutch"
        elif precond == "BB":
            update_precond, constr_precond = "sr1", "sr1"
        elif precond == "DB":
            update_precond, constr_precond = "hutch", "sr1"

        result = train_psps2(seed, train_data, train_target, train_dataloader, loss_class, EPOCHS, update_precond, 
                                   constr_precond, slack_method)
        
    elif optimizer_class == "sp2plus":
        result = train_sp2plus(seed, train_data, train_target, train_dataloader, loss_class, EPOCHS, slack_method)


    if save:
        results_path = os.getenv("RESULTS_DIR")
        directory = f"{results_path}/{dataset_name}/percent_{percentage}/scale_{scale}/bs_{batch_size}" \
        f"/epochs_{EPOCHS}/{loss_name}/{optimizer_class}/lr_{lr}/precond_{preconditioner}/slack_{slack_method}/lmd_{lmd}/mu_{mu}/seed_{seed}"
        print(directory)
        if not os.path.exists(directory):
            os.makedirs(directory)
            
        torch.save([x[0] for x in result], f"{directory}/loss")
        torch.save([x[1] for x in result], f"{directory}/grad_norm_sq")

        if slack_method in ("L1", "L2"):
            torch.save([x[2] for x in result], f"{directory}/slack")


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Help me!")
    parser.add_argument("--dataset", type=str, help="Name of a dataset from datasets directory.")
    parser.add_argument("--percent", type=restricted_float, default=1.0, help="What percentage of data to use. Range from (0.0, 1.0].")
    parser.add_argument("--scale", type=int, default=0, help="Scaling range. [-scale, scale].")
    parser.add_argument("--batch_size", type=int)
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--loss", type=str, choices=["logreg", "nllsq", "mse"])
    parser.add_argument("--optimizer", type=str, choices=["psps", "sp2plus", "psps2_DD", "psps2_BB", "psps2_DB"])
    parser.add_argument("--lr", type=float, default=0.1)
    parser.add_argument("--preconditioner", type=str, choices=["none", "hutch"], default="none")
    parser.add_argument("--slack", type=str, choices=["none", "L1", "L2"], default="none")
    parser.add_argument("--lmd", type=float, default=0.01, help="Lambda parameter.")
    parser.add_argument("--mu", type=float, default=0.1, help="Mu parameter.")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--save", action=argparse.BooleanOptionalAction, default=False, help="Select to save the results of the run.")
    args = parser.parse_args()
    print(f"device: {device}")
    print(args)

    for seed in [0, 1, 2, 3, 4]:
        for scale in [0, 3, 5]:
            main(dataset_name=args.dataset, batch_size=args.batch_size, percentage=args.percent, scale=scale, loss_name=args.loss, 
            optimizer_class=args.optimizer, EPOCHS=args.epochs, lr=args.lr, preconditioner=args.preconditioner, slack_method=args.slack, 
            lmd=args.lmd, mu=args.mu, seed=seed, save=args.save)
    






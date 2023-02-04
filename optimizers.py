import torch
from torch.optim import Optimizer
from utils import Hutch

from torch.optim import SGD, Adam
from utils import solve

def get_optimizer(optimizer):
    if optimizer == "sps":
        return SPS
    elif optimizer == "sps2":
        return SPS2
    elif optimizer == "sgd":
        return SGD
    elif optimizer == "adam":
        return Adam


class SPS(Optimizer):

    def __init__(self, params, preconditioner="none", slack_method="none", lmd=0.01, mu=0.1):
        defaults = dict(s=0.0)
        self.lmd = lmd
        self.lmd_hat = 1.0 / (1.0 + self.lmd)
        
        if slack_method == "L1":
            self.update = self.update_L1
        elif slack_method == "L2":
            self.update = self.update_L2
        else:
            self.update = self.update_sps

        self.preconditioner = None
        if preconditioner == "hutch":
            self.preconditioner = Hutch()
            
        self.preconditioner_initialized = False

        # TO-DO: Think of something better
        self.replay_buffer = []

        super().__init__(params, defaults)


    def step(self, closure=None):
        for group in self.param_groups:
            weights = list(group["params"])
            
            loss = closure()
            grad = torch.autograd.grad(loss, weights, create_graph=True)
            loss = loss.item()

            if self.preconditioner:
                if not self.preconditioner_initialized:
                    self.preconditioner.init(self, grad, 100)
                    self.preconditioner_initialized = True
                self.preconditioner.step(self, grad, 1)
            else:
                self.init_empty_precond()
                self.preconditioner_initialized = True
            
            gnorm_square = self.calc_grad_norm(grad).item()
            
            if gnorm_square < 1e-13:
                continue 

            self.update(gnorm_square, loss)

            self.replay_buffer.append({
                "loss": loss,
                "grad_norm_sq": gnorm_square,
                "slack": group['s'],
            }) 
                          
        return loss

    @torch.no_grad()
    def update_sps(self, gnorm_square, loss):
        for group in self.param_groups: 
            step_size = loss / gnorm_square
            for p in group['params']:
                p.sub_(self.state[p]['scaled_grad'].mul(step_size))
    

    @torch.no_grad()
    def update_L1(self, gnorm_square, loss):

            for group in self.param_groups:
                s = group['s']
                step_size_temp = torch.max(
                    torch.tensor(0.0), torch.tensor(loss - s + self.lmd) ) / (1 + gnorm_square)

                step_size = torch.min(step_size_temp, torch.tensor(loss / gnorm_square))

                group['s'] = torch.max(torch.tensor(0.0), 
                torch.tensor(s - self.lmd + step_size_temp))

                for p in group['params']:
                    p.sub_(self.state[p]['scaled_grad'].mul(step_size))

    @torch.no_grad()
    def update_L2(self, gnorm_square ,loss):
        
        for group in self.param_groups: 
            s = group['s']
            step_size = torch.max(torch.tensor(0.0), torch.tensor(loss - self.lmd_hat * s)) / (gnorm_square + self.lmd_hat)
            group['s'] = self.lmd_hat * (s + step_size)

            for p in group['params']:
                p.sub_(self.state[p]['scaled_grad'].mul(step_size))
            

    def init_empty_precond(self):
        for group in self.param_groups:
            for p in group['params']:
                self.state[p]['Dk'] = torch.ones_like(p) 
                self.state[p]['DkhatInv'] = torch.ones_like(p)

    def calc_grad_norm(self, grad):
        for group in self.param_groups: 
            gnorm_square = 0.
            for p, g in zip(group['params'], grad):
                g_detached = g.detach().clone()
                self.state[p]['scaled_grad'] = self.state[p]['DkhatInv'].mul(g_detached)
                gnorm_sq = self.state[p]['scaled_grad'].mul(g_detached).sum()
                gnorm_square += gnorm_sq

        return gnorm_square



class SPS2(Optimizer):

    def __init__(self, params, preconditioner="none", slack_method="none", lmd=0.01, mu=0.1):
        defaults = dict(s=0.0)
        self.lmd = lmd
        self.lmd_hat = 1.0 / (1.0 + self.lmd)
        self.s = 0.0
        self.z = 0.0
        self.mu = mu
        self.delta = lmd + mu
        
        if slack_method == "L1":
            self.update = self.update_L1
        elif slack_method == "L2":
            self.update = self.update_L2
        else:
            self.update = self.update_sps

        self.preconditioner = None
        if preconditioner == "hutch":
            self.preconditioner = Hutch()
            
        self.preconditioner_initialized = False

        # TO-DO: Think of something better
        self.replay_buffer = []

        super().__init__(params, defaults)


    def step(self, closure=None):
        for group in self.param_groups:
            weights = list(group["params"])
            
            loss = closure()
            grad = torch.autograd.grad(loss, weights, create_graph=True)
            loss = loss.item()

            if self.preconditioner:
                if not self.preconditioner_initialized:
                    self.preconditioner.init(self, grad, 100)
                    self.preconditioner_initialized = True
                self.preconditioner.step(self, grad, 1)
            else:
                self.init_empty_precond()
                self.preconditioner_initialized = True
            
            gnorm_square = self.calc_grad_norm(grad).item()
            
            if gnorm_square < 1e-13:
                continue 

            self.update(gnorm_square, loss)

            self.replay_buffer.append({
                "loss": loss,
                "grad_norm_sq": gnorm_square,
                "slack": self.s,
                "z": self.z # z = s - s_nil
            }) 
                          
        return loss

    @torch.no_grad()
    def update_sps(self, gnorm_square, loss):

        det = 1 - (2 * loss / gnorm_square )
        if det < 1e-15:
            return
        else:
            t = torch.sqrt(det)/det
            root1 = -1 + t
            root2 = -1 - t
            root = torch.maximum(root1, root2)
            precond = root/(1 + root)

        for group in self.param_groups: 
            for p in group['params']:
                p.sub_(self.state[p]['scaled_grad'].mul(precond))
    

    @torch.no_grad()
    def update_L1(self, gnorm_square, loss):

            s_nil = self.s - (self.lmd/(2*self.mu))
            self.z = self.s - s_nil

            t = loss - s_nil
            AA = 1
            BB = 2 + self.mu * gnorm_square - 2 * self.mu * t
            CC = 1 + 2 * self.mu * gnorm_square - 4 * self.mu * t
            DD = -2 * self.mu * t

            roots = solve(AA, BB, CC, DD)
            roots = torch.from_numpy(roots)
            root_star = torch.relu(torch.max(roots))

            self.s = torch.relu(self.s - ((self.lmd - root_star)/2 * self.mu)).item()
            
            precond = root_star/(1 + root_star)

            for group in self.param_groups:
                for p in group['params']:
                    p.sub_(self.state[p]['scaled_grad'].mul(precond))

    @torch.no_grad()
    def update_L2(self, gnorm_square ,loss):
        s_nil = self.s - (self.lmd/(2*self.mu))
        self.z = self.s - s_nil

        t = loss - s_nil

        AA = 1
        BB = 2 + self.delta * gnorm_square - 2 * self.delta * t
        CC = 1 + 2 * self.delta * gnorm_square - 4 * self.delta * t
        DD = -2 * self.delta * t

        roots = solve(AA, BB, CC, DD)
        roots = torch.from_numpy(roots)       
        root_star = torch.relu(torch.max(roots))

        self.s = (1/(self.delta)) * (self.mu * self.s + (root_star/2)).item()

        precond = root_star/(1 + root_star) 
        
        for group in self.param_groups: 
            for p in group['params']:
                p.sub_(self.state[p]['scaled_grad'].mul(precond))
            

    def init_empty_precond(self):
        for group in self.param_groups:
            for p in group['params']:
                self.state[p]['Dk'] = torch.ones_like(p) 
                self.state[p]['DkhatInv'] = torch.ones_like(p)

    def calc_grad_norm(self, grad):
        for group in self.param_groups: 
            gnorm_square = 0.
            for p, g in zip(group['params'], grad):
                g_detached = g.detach().clone()
                self.state[p]['scaled_grad'] = self.state[p]['DkhatInv'].mul(g_detached)
                gnorm_sq = self.state[p]['scaled_grad'].mul(g_detached).sum()
                gnorm_square += gnorm_sq

        return gnorm_square
import torch
from torch.optim import Optimizer
from utils import Hutch
from utils import solve

class PSPS(Optimizer):

    def __init__(self, params, preconditioner="none", slack_method="none", lmd=0.01, mu=0.1):
        defaults = dict(s=0.0, step_size=0.0)
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
                "step_size": group["step_size"],
            }) 
                          
        return loss

    @torch.no_grad()
    def update_sps(self, gnorm_square, loss):
        for group in self.param_groups: 
            group["step_size"] = loss / gnorm_square
            for p in group['params']:
                p.sub_(self.state[p]['scaled_grad'].mul(group["step_size"]))
    

    @torch.no_grad()
    def update_L1(self, gnorm_square, loss):

            for group in self.param_groups:
                s = group['s']
                step_size_temp = torch.max(
                    torch.tensor(0.0), torch.tensor(loss - s + self.lmd) ) / (1 + gnorm_square)

                group["step_size"] = torch.min(step_size_temp, torch.tensor(loss / gnorm_square))

                group['s'] = torch.max(torch.tensor(0.0), 
                torch.tensor(s - self.lmd + step_size_temp))

                for p in group['params']:
                    p.sub_(self.state[p]['scaled_grad'].mul(group["step_size"]))

    @torch.no_grad()
    def update_L2(self, gnorm_square ,loss):
        
        for group in self.param_groups: 
            s = group['s']
            group["step_size"] = torch.max(torch.tensor(0.0), torch.tensor(loss - self.lmd_hat * s)) / (gnorm_square + self.lmd_hat)
            group['s'] = self.lmd_hat * (s + group["step_size"])

            for p in group['params']:
                p.sub_(self.state[p]['scaled_grad'].mul(group["step_size"]))
            

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



class PSPS2(Optimizer):

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

        det = torch.tensor(1 - (2 * loss / gnorm_square ))
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




class PSPS2_B(Optimizer):

    def __init__(self, params, preconditioner="none", slack_method="none", s_method="sphere", lmd=0.01, mu=0.1):
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

        if s_method == "sphere":
            self.s_method = self.s_sphere
        elif s_method == "grad":
            self.s_method = self.s_grad
        elif s_method == "dgrad":
            self.s_method = self.s_dgrad

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
            
            # if gnorm_square < 1e-13:
            #     continue 

            self.init_B(grad)

            self.update(grad, loss)

            self.replay_buffer.append({
                "loss": loss,
                "grad_norm_sq": gnorm_square,
                "slack": self.s,
                "z": self.z # z = s - s_nil
            }) 
                          
        return loss


    @torch.no_grad()
    def s_sphere(self, **kwargs):
        shape = kwargs["shape"]
        s = torch.randn(shape)
        s.div_(s.norm())
        return s

    @torch.no_grad()
    def s_grad(self, **kwargs):
        grad = kwargs["grad"]
        return grad

    @torch.no_grad()
    def s_dgrad(self, **kwargs):
        p = kwargs["param"]
        grad = kwargs["grad"]
        return self.state[p]['DkhatInv'] * grad


    @torch.no_grad()
    def init_B(self, grad):
        for group in self.param_groups:
            for p, g in zip(group["params"], grad):
                # sk = self.s_sphere(p.shape[0]).to(p.device)
                
                sk = self.s_method(shape=p.shape[0], grad=g, param=p).to(p.device)

                yk = self.state[p]['DkhatInv'] * sk
                yk2 = torch.square(self.state[p]['DkhatInv']) * sk

                self.state[p]["D_inv_mx"] =  torch.diagflat(self.state[p]['DkhatInv'].clone().detach())
                self.state[p]["D_inv_B"] = ((yk2.reshape(-1, 1) @ yk.reshape(1, -1)) / (yk.dot(sk)))
                self.state[p]['D_inv_B_D_inv'] = self.state[p]["D_inv_B"] * self.state[p]['DkhatInv']


    @torch.no_grad()
    def update_sps(self, grad, loss):

        for group in self.param_groups: 
            for p, g in zip(group['params'], grad):

                Dk_hat_inv = self.state[p]['DkhatInv']
                D_inv_B_D_inv = self.state[p]['D_inv_B_D_inv']
                D_inv_B = self.state[p]["D_inv_B"]
                D_inv = self.state[p]["D_inv_mx"]

                a = torch.dot(g, Dk_hat_inv*g)
                a = a.cpu().detach().numpy()

                b = torch.dot(g, D_inv_B_D_inv@g)
                b = b.cpu().detach().numpy() 

                c_torch = torch.trace(D_inv_B)
                c = c_torch.cpu().detach().numpy()

                AA = 2 * a * c ** 2
                BB = 4 * a * c - 2 * loss * c**2 - b * c
                CC = 2 * a - 4 * c * loss - 2 * b
                DD = - 2 * loss

                BB = BB / AA
                CC = CC / AA
                DD = DD / AA

                AA = torch.tensor(1.0)


                lmds = solve(AA, BB, CC, DD)
                lmds = torch.from_numpy(lmds).to(p.device)
                lmd_max = torch.max(lmds)
                lmd_min = torch.maximum(torch.min(lmds), torch.tensor(0))
                lmd_star = lmd_max

                if self.lagr(lmd_max, p, loss, g) < self.lagr(lmd_min, p, loss, g):
                    lmd_star = lmd_min

                precond = lmd_star * ( D_inv - (lmd_star * (D_inv_B_D_inv) / (1 + lmd_star * c_torch)) )

                p.sub_(precond @ g)
    

    @torch.no_grad()
    def update_L1(self, grad, loss):

            for group in self.param_groups:
                for p, g in zip(group['params'], grad):

                    Dk_hat_inv = self.state[p]['DkhatInv']
                    D_inv_B_D_inv = self.state[p]['D_inv_B_D_inv']
                    D_inv_B = self.state[p]["D_inv_B"]
                    D_inv = self.state[p]["D_inv_mx"]

                    s_nil = self.s - self.lmd/(2 * self.mu)
                    t = loss - s_nil

                    a = torch.dot(g, Dk_hat_inv * g)
                    a = a.cpu().detach().numpy()

                    b = torch.dot(g, D_inv_B_D_inv@g)
                    b = b.cpu().detach().numpy() 

                    c_torch = torch.trace(D_inv_B)
                    c = c_torch.cpu().detach().numpy()

                    d = 1 + 2 * self.mu * a

                    AA = d * c**2
                    BB = 2 * c * d - 2 * self.mu * t * c**2 - b * c * self.mu
                    CC = d - 4 * self.mu * t * c - 2 * b * self.mu
                    DD = -2 * self.mu * t

                    BB = BB / AA
                    CC = CC / AA
                    DD = DD / AA
                    AA = torch.tensor(1.0)

                    lmds = solve(AA, BB, CC, DD)
                    lmds = torch.from_numpy(lmds).to(p.device)
                        
                    lmd_max = torch.max(lmds)
                    lmd_star = torch.maximum(lmd_max, torch.tensor(0.0))
                    lmd_min = torch.maximum(torch.min(lmds), torch.tensor(0.0))
                    lmd_star = lmd_max
                    
                    if self.lagr(lmd_max, p, loss, g) < self.lagr(lmd_min, p, loss, g):
                        lmd_star = lmd_min
            
                    self.s = torch.maximum(torch.tensor(0.0), self.s - 1/(2*self.mu)*(self.lmd - lmd_star)).item()

                    precond = lmd_star * ( D_inv - (lmd_star * (D_inv_B_D_inv) / (1 + lmd_star * c_torch)) )

                    p.sub_(precond @ g)

    @torch.no_grad()
    def update_L2(self, grad ,loss):
        
        for group in self.param_groups: 
            for p, g in zip(group['params'], grad):

                Dk_hat_inv = self.state[p]['DkhatInv']
                D_inv_B_D_inv = self.state[p]['D_inv_B_D_inv']
                D_inv_B = self.state[p]["D_inv_B"]
                D_inv = self.state[p]["D_inv_mx"]

                
                s_nil = self.s - self.lmd/(2*self.mu)
                t = loss - s_nil

                a = torch.dot(g, Dk_hat_inv*g)
                a = a.cpu().detach().numpy()

                b = torch.dot(g, D_inv_B_D_inv@g)
                b = b.cpu().detach().numpy() 

                c_torch = torch.trace(D_inv_B)
                c = c_torch.cpu().detach().numpy()

                d = a + 1/(2 * self.delta) 

                AA = 2 * d * c**2
                BB = 4 * c * d - 2 * t * c**2 - b * c
                CC = 2 * d - 4 * t * c - 2 * b
                DD = -2 * t

                BB = BB / AA
                CC = CC / AA
                DD = DD / AA

                AA = torch.tensor(1.0)

                lmds = solve(AA, BB, CC, DD)
                lmds = torch.from_numpy(lmds).to(p.device)

                lmd_max = torch.max(lmds)
                lmd_star = torch.maximum(lmd_max, torch.tensor(0.0))
                lmd_min = torch.maximum(torch.min(lmds), torch.tensor(0.0))
                lmd_star = lmd_max
                
                if self.lagr(lmd_max, p, loss, g) < self.lagr(lmd_min, p, loss, g):
                    lmd_star = lmd_min

                self.s = (1/self.delta * (self.mu * self.s + lmd_star/2)).item()

                precond = lmd_star * ( D_inv - (lmd_star * (D_inv_B_D_inv) / (1 + lmd_star * c_torch)) )

                p.sub_(precond @ g)
            

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


    @torch.no_grad()
    def lagr(self, lmd, param, loss, grad):
        D_inv = self.state[param]["D_inv_mx"]
        D_inv_B_D_inv = self.state[param]["D_inv_B_D_inv"]
        c = torch.trace(self.state[param]["D_inv_B"])
        b = lambda l: D_inv - (l * (D_inv_B_D_inv) / (1 + l * c)) 
        return lmd * loss -  (1/2)*lmd**2 * torch.dot(grad, b(lmd)@grad)
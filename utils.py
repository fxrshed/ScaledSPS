import torch
import argparse

def rademacher(weights):
    return [torch.round(torch.rand_like(w)) * 2 - 1 for w in weights]

def hvp_from_grad(grads_tuple, list_params, vec_tuple):
    # don't damage grads_tuple. Grads_tuple should be calculated with create_graph=True
    dot = 0.
    for grad, vec in zip(grads_tuple, vec_tuple):
        dot += grad.mul(vec).sum()
    return torch.autograd.grad(dot, list_params, retain_graph=True)


class Hutch(object):

    def __init__(self):
        self.beta = 0.999
        self.alpha = 0.1

    def init(self, model, grad, iters=100):
        for group in model.param_groups:
            weights = list(group["params"])
            
            Dk = self.diag_estimate(weights, grad, iters)
            for p, Dki in zip(group['params'], Dk):
                model.state[p]['Dk'] = Dki 
        
    def step(self, model, grad, iters):
        for group in model.param_groups:
            weights = list(group["params"])
            
            vk = self.diag_estimate(weights, grad, iters)
            
            # Smoothing and Truncation 
            with torch.no_grad():
                for p, v in zip(group['params'], vk):
                    model.state[p]['Dk'].mul_(self.beta).add_(v, alpha = 1 - self.beta)
                    model.state[p]['DkhatInv'] = torch.reciprocal(torch.clamp(torch.abs(model.state[p]['Dk']), min = self.alpha))
            

    def diag_estimate(self, weights, grad, iters):
        estimates = [torch.zeros_like(w) for w in weights]
        for j in range(iters):
            rand_rad = rademacher(weights)
            hvp = hvp_from_grad(grad, weights, rand_rad)
            with torch.no_grad():
                for r, p, es in zip(rand_rad, hvp, estimates):
                    es.mul_(j/(j+1))
                    es.add_(p.detach().mul(r).div(j+1))
        return estimates



def restricted_float(x):
    try:
        x = float(x)
    except ValueError:
        raise argparse.ArgumentTypeError("%r not a floating-point literal" % (x,))

    if x < 0.01 or x > 1.0:
        raise argparse.ArgumentTypeError("%r not in range [0.01, 1.0]"%(x,))
    return x

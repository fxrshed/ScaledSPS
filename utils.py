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


# CUBIC ROOT SOLVER

# Date Created   :    24.05.2017
# Created by     :    Shril Kumar [(shril.iitdhn@gmail.com),(github.com/shril)] &
#                     Devojoyti Halder [(devjyoti.itachi@gmail.com),(github.com/devojoyti)]

# Project        :    Classified 
# Use Case       :    Instead of using standard numpy.roots() method for finding roots,
#                     we have implemented our own algorithm which is ~10x faster than
#                     in-built method.

# Algorithm Link :    www.1728.org/cubic2.htm

# This script (Cubic Equation Solver) is an independent program for computation of roots of Cubic Polynomials. This script, however,
# has no relation with original project code or calculations. It is to be also made clear that no knowledge of it's original project 
# is included or used to device this script. This script is complete freeware developed by above signed users, and may further be
# used or modified for commercial or non-commercial purpose.


# Libraries imported for fast mathematical computations.
import math
import numpy as np

# Main Function takes in the coefficient of the Cubic Polynomial
# as parameters and it returns the roots in form of numpy array.
# Polynomial Structure -> ax^3 + bx^2 + cx + d = 0

def solve(a, b, c, d):

    if (a == 0 and b == 0):                     # Case for handling Liner Equation
        return np.array([(-d * 1.0) / c])                 # Returning linear root as numpy array.

    elif (a == 0):                              # Case for handling Quadratic Equations

        D = c * c - 4.0 * b * d                       # Helper Temporary Variable
        if D >= 0:
            D = math.sqrt(D)
            x1 = (-c + D) / (2.0 * b)
            x2 = (-c - D) / (2.0 * b)

            return np.array([x1, x2])
        else:
            D = math.sqrt(-D)
            x1 = (-c + D * 1j) / (2.0 * b)
            x2 = (-c - D * 1j) / (2.0 * b)

            return np.array([0.0, 0.0])
            
        # return np.array([x1, x2])               # Returning Quadratic Roots as numpy array.

    f = findF(a, b, c)                          # Helper Temporary Variable
    g = findG(a, b, c, d)                       # Helper Temporary Variable
    h = findH(g, f)                             # Helper Temporary Variable

    if f == 0 and g == 0 and h == 0:            # All 3 Roots are Real and Equal
        if (d / a) >= 0:
            x = (d / (1.0 * a)) ** (1 / 3.0) * -1
        else:
            x = (-d / (1.0 * a)) ** (1 / 3.0)

        return np.array([x, x, x])              # Returning Equal Roots as numpy array.

    elif h <= 0:                                # All 3 roots are Real

        i = math.sqrt(((g ** 2.0) / 4.0) - h)   # Helper Temporary Variable
        j = i ** (1 / 3.0)                      # Helper Temporary Variable
        k = math.acos(-(g / (2 * i)))           # Helper Temporary Variable
        L = j * -1                              # Helper Temporary Variable
        M = math.cos(k / 3.0)                   # Helper Temporary Variable
        N = math.sqrt(3) * math.sin(k / 3.0)    # Helper Temporary Variable
        P = (b / (3.0 * a)) * -1                # Helper Temporary Variable

        x1 = 2 * j * math.cos(k / 3.0) - (b / (3.0 * a))
        x2 = L * (M + N) + P
        x3 = L * (M - N) + P

        return np.array([x1, x2, x3])           # Returning Real Roots as numpy array.

    elif h > 0:                                 # One Real Root and two Complex Roots
        R = -(g / 2.0) + math.sqrt(h)           # Helper Temporary Variable
        if R >= 0:
            S = R ** (1 / 3.0)                  # Helper Temporary Variable
        else:
            S = (-R) ** (1 / 3.0) * -1          # Helper Temporary Variable
        T = -(g / 2.0) - math.sqrt(h)
        if T >= 0:
            U = (T ** (1 / 3.0))                # Helper Temporary Variable
        else:
            U = ((-T) ** (1 / 3.0)) * -1        # Helper Temporary Variable

        x1 = (S + U) - (b / (3.0 * a))
        x2 = -(S + U) / 2 - (b / (3.0 * a)) + (S - U) * math.sqrt(3) * 0.5j
        x3 = -(S + U) / 2 - (b / (3.0 * a)) - (S - U) * math.sqrt(3) * 0.5j

        # return np.array([x1, x2, x3])           # Returning One Real Root and two Complex Roots as numpy array.
        return np.array([x1, 0, 0])


# Helper function to return float value of f.
def findF(a, b, c):
    return ((3.0 * c / a) - ((b ** 2.0) / (a ** 2.0))) / 3.0


# Helper function to return float value of g.
def findG(a, b, c, d):
    return (((2.0 * (b ** 3.0)) / (a ** 3.0)) - ((9.0 * b * c) / (a **2.0)) + (27.0 * d / a)) /27.0


# Helper function to return float value of h.
def findH(g, f):
    return ((g ** 2.0) / 4.0 + (f ** 3.0) / 27.0)
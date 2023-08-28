import torch


def logreg(w, X, y):
    return torch.mean(torch.log(1 + torch.exp(-y * (X @ w))))

def grad_logreg(w, X, y):
    r = torch.exp(-y * (X @ w))
    return ( (r/(1 + r)) @ (X * -y[:, None]) ) / X.shape[0]

def hess_logreg(w, X, y):
    r = torch.exp(-y * (X @ w))
    return ( X.T @ (  (r/torch.square(1 + r)).reshape(-1, 1) * X ) ) / X.shape[0]


def nllsq(w, X, y):
    return torch.mean( ( y - (1/(1 + torch.exp(-X @ w ))) )**2 )


def mse(w, X, y):
    return 0.5 * ( torch.sum(torch.square(X @ w - y))  ) / X.shape[0]

def grad_mse(w, X, y):
    return (X.T @ (X @ w - y)) / X.shape[0]

def hess_mse(w, X, y):
    return  (X.T @ X) / X.shape[0]

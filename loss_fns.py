import torch

def logistic_reg(w, X, y):
    return torch.mean(torch.log(1 + torch.exp(-y * (X @ w))))

def nllsq(w, X, y):
    return torch.mean( ( y - (1/(1 + torch.exp(-X @ w ))) )**2 )


def get_loss(loss):
    if loss == "logreg":
        return LogisticRegression
    elif loss == "nllsq":
        return NLLSQ
    elif loss == "nll_loss":
        return torch.functional.F.nll_loss
    else:
        raise ValueError("Non-existent loss requested.")

class LogisticRegression(torch.nn.Module):
    y_range = torch.tensor([-1., 1.])
    def __init__(self, params):
        self.params = params
        super().__init__()

    def forward(self, input, target):
        return torch.mean(torch.log(1 + torch.exp(-target * (input @ self.params))))


class NLLSQ(torch.nn.Module):
    y_range = torch.tensor([0., 1.])
    def __init__(self, params):
        self.params = params
        super().__init__()

    def forward(self, input, target):
        return torch.mean((target - (1/(1 + torch.exp(-input @ self.params))))**2)


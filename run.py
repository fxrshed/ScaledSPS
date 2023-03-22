import os 
import argparse

import torch
from torch.utils.data import DataLoader
import torch.utils.data as data_utils
from torch.optim import SGD, Adam, Adagrad, Adadelta, RMSprop
from torch_optimizer import Adahessian

from datasets import get_dataset
from loss_fns import get_loss
from optimizers import PSPS, PSPS2, PSPS2_B

from utils import restricted_float

from torch.utils.tensorboard import SummaryWriter

from dotenv import load_dotenv
load_dotenv()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


optimizers_dict = {
    "psps": PSPS,
    "psps2": PSPS2,
    "psps2_b": PSPS2_B,
    "sgd": SGD,
    "adam": Adam,
    "adagrad": Adagrad,
    "adadelta": Adadelta,
    "rmsprop": RMSprop,
    "adahessian": Adahessian,
}

psps_classes = (PSPS, PSPS2, PSPS2_B)


def train(seed, criterion_class, train_data, train_target, batch_size, steps, optimizer_class, tb, **optimizer_kwargs):
    
    torch.random.manual_seed(seed)

    params = torch.zeros(train_data.shape[1], device=device).requires_grad_()

    train_load = data_utils.TensorDataset(train_data, train_target)
    train_dataloader = DataLoader(train_load, batch_size=batch_size, shuffle=True)

    criterion = criterion_class(params)
    optimizer = optimizer_class([params], **optimizer_kwargs)

    train_loss = criterion(train_data.to(device), train_target.to(device))
    g, = torch.autograd.grad(train_loss, params)
    grad_norm_sq = torch.linalg.norm(g).item() ** 2
    slack = 0

    if tb:
        tb.add_scalar("loss", train_loss, 0)
        tb.add_scalar("grad_norm_squared", grad_norm_sq, 0)
        if isinstance(optimizer, psps_classes):
            tb.add_scalar("slack", slack, 0)


    hist = [[train_loss.item(), grad_norm_sq, slack]]
   
    for step in range(steps):
        for i, (batch_data, batch_target) in enumerate(train_dataloader):  
            batch_data = batch_data.to(device)
            batch_target = batch_target.to(device)

            optimizer.zero_grad()
            
            def closure():
                return criterion(batch_data, batch_target)
            
            loss = closure()

            if isinstance(optimizer, psps_classes):     
                optimizer.step(closure) 
                slack = optimizer.replay_buffer[-1]["slack"]
                if tb:
                    tb.add_scalar("slack", slack, step + 1)
            elif isinstance(optimizer, Adahessian): 
                loss.backward(create_graph=True)    
                optimizer.step()
            else:
                loss.backward()
                optimizer.step()

        train_loss = criterion(train_data.to(device), train_target.to(device))
        g, = torch.autograd.grad(train_loss, params)
        grad_norm_sq = torch.linalg.norm(g).item() ** 2

        if tb:
            tb.add_scalar("loss", train_loss, step + 1)
            tb.add_scalar("grad_norm_squared", grad_norm_sq, step + 1)

        hist.append([train_loss.item(), grad_norm_sq, slack])
    
        print(f"Loss: {train_loss.item()} | GradNorm^2: {grad_norm_sq}")

    return hist


def main(dataset, percent, scale, batch_size, epochs, 
        loss_class, optimizer_class, lr, preconditioner, 
        slack_method, lmd, mu, seed, save, tb):


    if tb:
        comment = f"/{dataset}-Percent:{percent}-Scaled:[{-scale},{scale}]-BatchSize:{batch_size}-" \
            f"Epochs:{epochs}-Loss:{loss_class}-Optimizer:{optimizer_class}-" \
            f"Lr:{lr}-Precond:{preconditioner}-Slack:{slack_method}-"\
            f"Lmd:{lmd}-Mu:{mu}-Seed:{seed}"
        tb_writer = SummaryWriter(comment=comment)
    else:
        tb_writer = None

    loss = get_loss(loss_class)
    optimizer = optimizers_dict[optimizer_class]
    scale_range = [-scale, scale]
    train_data, train_target = get_dataset(dataset, batch_size, percent, scale_range, loss.y_range) 


    if optimizer_class in ("sgd", "adam", "adagrad", "adadelta", "rmsprop", "adahessian"):
        result = train(
            seed,
            loss,
            train_data, 
            train_target, 
            batch_size,
            epochs,
            optimizer,
            tb_writer,
            lr=lr
        )
    elif  optimizer_class in ("psps", "psps2", "psps2_b"):
       result = train(
            seed,
            loss,
            train_data, 
            train_target, 
            batch_size,
            epochs,
            optimizer,
            tb_writer,
            preconditioner=preconditioner,
            slack_method=slack_method,
            lmd=lmd,
            mu=mu
        ) 

    if tb:
        tb_writer.flush()
        tb_writer.close()

    if save:
        results_path = os.getenv("RESULTS_DIR")
        directory = f"{results_path}/{dataset}/percent_{percent}/scale_{scale}/bs_{batch_size}" \
        f"/epochs_{epochs}/{loss_class}/{optimizer_class}/lr_{lr}/precond_{preconditioner}/slack_{slack_method}/lmd_{lmd}/mu_{mu}/seed_{seed}"
        print(directory)
        if not os.path.exists(directory):
            os.makedirs(directory)
            
        torch.save([x[0] for x in result], f"{directory}/loss")
        torch.save([x[1] for x in result], f"{directory}/grad_norm_sq")
        
        if optimizer_class in ("psps", "psps2", "psps2_b"):
            torch.save([x[2] for x in result], f"{directory}/slack")



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Help me!")
    parser.add_argument("--dataset", type=str, help="Name of a dataset from datasets directory.")
    parser.add_argument("--percent", type=restricted_float, default=1.0, help="What percentage of data to use. Range from (0.0, 1.0].")
    parser.add_argument("--scale", type=int, default=0, help="Scaling range. [-scale, scale].")
    parser.add_argument("--batch_size", type=int)
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--loss", type=str, choices=["logreg", "nllsq"])
    parser.add_argument("--optimizer", type=str, choices=["psps", "psps2", "psps2_b", "sgd", "adam", "adagrad", "adadelta", "rmsprop", "adahessian"])
    parser.add_argument("--lr", type=float, default=0.1)
    parser.add_argument("--preconditioner", type=str, choices=["none", "hutch"], default="none")
    parser.add_argument("--slack", type=str, choices=["none", "L1", "L2"], default="none")
    parser.add_argument("--lmd", type=float, default=0.01, help="Lambda parameter.")
    parser.add_argument("--mu", type=float, default=0.1, help="Mu parameter.")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--save", action=argparse.BooleanOptionalAction, default=False, help="Select to save the results of the run.")
    parser.add_argument("--tb", action=argparse.BooleanOptionalAction, help="Select to log metrics to Tensorboard.") 

    args = parser.parse_args()

    print(args)

    main(args.dataset, args.percent, args.scale, args.batch_size, args.epochs, args.loss, args.optimizer, args.lr,
    args.preconditioner, args.slack, args.lmd, args.mu, args.seed, args.save, args.tb)
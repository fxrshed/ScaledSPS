import datetime
import os 
import argparse
import socket

import torch

from datasets import get_dataset
from loss_fns import get_loss
from optimizers import get_optimizer, SPS
from nn_models import get_model

from torch.utils.tensorboard import SummaryWriter

from dotenv import load_dotenv
load_dotenv()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

@torch.no_grad()
def eval_model(model, loss_fn, data_loader):
    n_correct = 0
    n_samples = 0
    loss = 0
    for images, labels in data_loader:
        images = images.to(device)
        labels = labels.to(device)
        outputs = model(images)
        loss += loss_fn(outputs, labels).item() / len(data_loader)
        # max returns (value ,index)
        _, predicted = torch.max(outputs.data, 1)
        n_samples += labels.size(0)
        n_correct += (predicted == labels).sum().item()  
    
    acc = 100.0 * n_correct / n_samples

    return loss, acc


def train_nn(model, criterion, train_loader, test_loader, epochs, optimizer_class, tb, **optimizer_kwargs):

    optimizer = optimizer_class(model.parameters(), **optimizer_kwargs)

    slack = 0

    hist = []
    with torch.no_grad():
        train_loss, train_acc = eval_model(model, criterion, train_loader) 
        print(f"Epoch[{0}] Train Loss: {train_loss} | Train Acc: {train_acc}")

        test_loss, test_acc = eval_model(model, criterion, test_loader)
        print(f"Epoch[{0}] Test Loss: {test_loss} | Test Acc: {test_acc}")

        if tb:
            tb.add_scalar("train/loss", train_loss, 0)
            tb.add_scalar("train/acc", train_acc, 0)
            tb.add_scalar("test/loss", test_loss, 0)
            tb.add_scalar("test/acc", test_acc, 0)
            if isinstance(optimizer, SPS):
                tb.add_scalar("slack", slack, 0)

        hist.append([train_loss, train_acc, test_loss, test_acc, slack])
        
    for epoch in range(epochs):
        for i, (images, labels) in enumerate(train_loader):  
            
            # images = images.reshape(-1, images.shape[2] * images.shape[3]).to(device)
            images = images.to(device)
            labels = labels.to(device)
            
            optimizer.zero_grad()
            
            def closure():
                outputs = model(images)
                return criterion(outputs, labels)
            
            loss = closure()

            if isinstance(optimizer, SPS):     
                optimizer.step(closure) 
                slack = optimizer.replay_buffer[-1]["slack"]
            else:
                loss.backward()
                optimizer.step()

        if epoch % 10 == 0:
            with torch.no_grad():
                train_loss, train_acc = eval_model(model, criterion, train_loader) 
                print(f"Epoch[{epoch + 1}] Train Loss: {train_loss} | Train Acc: {train_acc}")

                test_loss, test_acc = eval_model(model, criterion, test_loader)
                print(f"Epoch[{epoch + 1}] Test Loss: {test_loss} | Test Acc: {test_acc}")

                if tb:
                    tb.add_scalar("train/loss", train_loss, epoch + 1)
                    tb.add_scalar("train/acc", train_acc, epoch + 1)
                    tb.add_scalar("test/loss", test_loss, epoch + 1)
                    tb.add_scalar("test/acc", test_acc, epoch + 1)
                    if isinstance(optimizer, SPS):
                        tb.add_scalar("slack", slack, epoch + 1)

                hist.append([train_loss, train_acc, test_loss, test_acc, slack])
        
    return hist


def main(dataset, model_class, batch_size, epochs, 
        loss_class, optimizer_class, lr, preconditioner, 
        slack_method, lmd, seed, save, tb):


    if tb:
        comment = f"/{dataset}-Model:{model_class}-BatchSize:{batch_size}-" \
            f"Epochs:{epochs}-Loss:{loss_class}-Optimizer:{optimizer_class}-" \
            f"Lr:{lr}-Precond:{preconditioner}-Slack:{slack_method}-"\
            f"Lmd:{lmd}-Seed:{seed}"   
        current_time = datetime.datetime.now().strftime("%b%d_%H-%M-%S")
        log_dir = os.path.join(
            "nn_runs", current_time + "_" + socket.gethostname() + comment
        )
         
        tb_writer = SummaryWriter(log_dir=log_dir)
    else:
        tb_writer = None

    torch.manual_seed(seed)

    model = get_model(model_class)().to(device)
    loss = get_loss(loss_class)
    optimizer = get_optimizer(optimizer_class)
    train_loader, test_loader = get_dataset(dataset, batch_size) 


    if optimizer_class in ["sgd", "adam"]:
        result = train_nn(
                    model, 
                    loss,
                    train_loader,
                    test_loader,
                    epochs,
                    optimizer,
                    tb_writer,
                    lr=lr
                )
    elif optimizer_class == "sps":
       result = train_nn(
                    model, 
                    loss,
                    train_loader,
                    test_loader,
                    epochs,
                    optimizer,
                    tb_writer,
                    preconditioner=preconditioner,
                    slack_method=slack_method,
                    lmd=lmd
                )

    if tb:
        tb_writer.flush()
        tb_writer.close()

    if save:
        results_path = os.getenv("RESULTS_DIR")
        directory = f"{results_path}/{model_class}/{dataset}/bs_{batch_size}/epochs_{epochs}/{loss_class}/{optimizer_class}/lr_{lr}/precond_{preconditioner}/slack_{slack_method}/lmd_{lmd}/seed_{seed}"
        print(directory)
        if not os.path.exists(directory):
            os.makedirs(directory)
            
        torch.save([x[0] for x in result], f"{directory}/train_loss")
        torch.save([x[1] for x in result], f"{directory}/train_acc")
        torch.save([x[2] for x in result], f"{directory}/test_loss")
        torch.save([x[3] for x in result], f"{directory}/test_acc") 
        
        if optimizer_class == "sps":
            torch.save([x[4] for x in result], f"{directory}/slack")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Help me!")
    parser.add_argument("--dataset", type=str, choices=["MNIST"])
    parser.add_argument("--model", type=str, choices=["smlenet"], default="smlenet")
    parser.add_argument("--batch_size", type=int)
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--loss", type=str, choices=["nll_loss"])
    parser.add_argument("--optimizer", type=str, choices=["sgd", "sps", "adam"])
    parser.add_argument("--lr", type=float, default=0.1)
    parser.add_argument("--preconditioner", type=str, choices=["none", "hutch"], default="none")
    parser.add_argument("--slack", type=str, choices=["none", "L1", "L2"], default="none")
    parser.add_argument("--lmd", type=float, default=0.01, help="Lambda parameter.")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--save", action=argparse.BooleanOptionalAction, default=False, help="Select to save the results of the run.")
    parser.add_argument("--tb", action=argparse.BooleanOptionalAction, help="Select to log metrics to Tensorboard.") 

    args = parser.parse_args()

    print(args)

    main(args.dataset, args.model, args.batch_size, args.epochs, args.loss, args.optimizer, args.lr,
    args.preconditioner, args.slack, args.lmd, args.seed, args.save, args.tb)

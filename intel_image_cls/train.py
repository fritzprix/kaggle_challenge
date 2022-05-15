from torchvision import datasets
from torch.utils import data
from typing import Tuple
import argparse
import torch
from torch import nn
from torchvision import transforms as T
import wandb
from tqdm import tqdm
import model

PRJ_NAME = 'intel_image_cls'

def accuracy(y_hat: torch.Tensor, y: torch.Tensor):
    if len(y_hat.shape) > 1 and y_hat.shape[1] > 1:
        y_hat = y_hat.argmax(axis=1)
    return float((y_hat.type(y.dtype) == y).sum())


def train_epoch(net, train_data, loss, updater, device):
    net.train()
    acc_sum, lsum, numel = 0, 0, 0
    for X, y in tqdm(train_data):
        X = X.to(device)
        y = y.to(device)
        updater.zero_grad()
        y_hat = net(X)
        l = loss(y_hat, y)
        l.mean().backward()
        updater.step()
        lsum += l.sum().detach()
        numel += y.numel()
        acc_sum += accuracy(y_hat, y)

    return (lsum / numel), (acc_sum / numel)

def init_weight(m):
    if type(m) == nn.Conv2d or type(m) == nn.Linear:
        nn.init.xavier_normal_(m.weight)
        print(f'initialize weight : {m}')

def train(net: nn.Module, train_data, val_data, device, config):
    with wandb.init(project=PRJ_NAME, job_type='training') as run:
        lr, num_epochs = config['lr'], config['epochs']
        net = net.to(device)
        net.apply(init_weight)
        print(f'network : {net.__class__.__name__}')
        
        loss = nn.CrossEntropyLoss()
        updater = torch.optim.SGD(net.parameters(), lr)
        loss = loss.to(device)
        for epoch in range(num_epochs):
            tloss, tacc = train_epoch(net, train_data, loss, updater, device)
            val_loss, val_acc = evaluate(net, val_data, loss, device)
            metric = {
                'train_loss': tloss,
                'train_accuracy': tacc,
                'validation_loss': val_loss,
                'validation_accuracy': val_acc
            }
            wandb.log(metric)
            print(f'Ep {epoch} : {metric}')

        torch.save(net.state_dict(), f'{net.__class__.__name__}.pt')
        trained_model = wandb.Artifact(f'{net.__class__.__name__}', type='model', description=f'{str(net)}')
        trained_model.add_file(f'{net.__class__.__name__}.pt')
        run.log_artifact(trained_model)

    
def evaluate(net, val_data, loss, device):
    net.eval()
    with torch.no_grad():
        acc_sum, lsum, numel = 0, 0, 0
        for X, y in val_data:
            X = X.to(device)
            y = y.to(device)
            y_hat = net(X)
            lsum += loss(y_hat, y).sum()
            acc_sum += accuracy(y_hat, y)
            numel += y.numel()
    return (lsum / numel), (acc_sum / numel)


def try_gpu():
    for i in range(torch.cuda.device_count()):
        return torch.device(f'cuda:{i}')
    return torch.device('cpu')


def main(args):
    

    wandb.init(project=PRJ_NAME, entity="dwidlee")
    config = wandb.config
    print(config)
    print(args)
    device = try_gpu()
    if 'lr' not in config:
        config['lr'] = args.lr
    if 'epochs' not in config:
        config['epochs'] = args.epochs
    if 'batch' not in config:
        config['batch'] = args.b
    
    print(config)

    validate_set = datasets.ImageFolder(root='data/seg_test', transform=T.Compose([T.ToTensor(), Padding((150, 150))]))
    train_set = datasets.ImageFolder(root='data/seg_train', transform=T.Compose([T.ToTensor(), Padding((150, 150))]))
    train_data = data.DataLoader(train_set, batch_size=config['batch'], shuffle=True, num_workers=5)
    val_data = data.DataLoader(validate_set, batch_size=config['batch'], shuffle=True, num_workers=5)

    train(model.DenseInception_V0(), train_data, val_data, device, config)
    


class Padding(object):
    
    def __init__(self, size: Tuple):
        super(Padding).__init__()
        self.size = size

    def __call__(self, X: torch.Tensor):
        ## torch compatible image fromat assumed (N,C,H,W)
        h, w = self.size
        if X.shape[1] == h and X.shape[2] == w:
            return X
        else:
            tp = (h - X.shape[1]) // 2
            bp = h - tp - X.shape[1]
            lp = (w - X.shape[2]) // 2
            rp = w - lp - X.shape[2]
            return T.Pad((lp, tp, rp, bp))(X)
        

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="train model")
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--lr', type=float, default=0.01)
    parser.add_argument('--b', type=int, default=4)
    parser.add_argument('--o', type=str)
    args = parser.parse_args()


    main(args)
    
from genericpath import exists, isdir
from pip import main
import wandb
import torch
from model import KLS
from model import DenseInception_V0
from model import Preprocessor
from torchvision import transforms as T
import argparse
from torchvision import datasets
from tqdm import tqdm
from torch.utils import data
from PIL import Image
from torch import nn
from model import Preprocessor
import csv

def accuracy(y_hat: torch.Tensor, y: torch.Tensor):
    if len(y_hat.shape) > 1 and y_hat.shape[1] > 1:
        y_hat = y_hat.argmax(axis=1)
    return float((y_hat.type(y.dtype) == y).sum())


def predict(net, data, device):
    net = net.to(device)
    result = []
    for X, _ in tqdm(data):
        X = X.to(device)
        y_hat = net(X)
        result.extend([int(i) for i in y_hat.argmax(axis=1)])
    return result

def validate(net, device):
    
    val_dataset = datasets.ImageFolder('data/seg_test', transform=Preprocessor())
    val_data = data.DataLoader(val_dataset, shuffle=True)
    net.eval()
    net = net.to(device)

    loss = nn.CrossEntropyLoss()
    loss = loss.to(device)
    lsum, count, acc = 0, 0, 0

    for X, y in tqdm(val_data):
        X = X.to(device)
        y = y.to(device)
        y_hat = net(X)
        l = loss(y_hat, y)
        lsum += float(l.sum())
        count += float(y.numel())
        acc += float(accuracy(y_hat, y))
    return (lsum / count), (acc / count)

        
def main(args):
    
        print(DenseInception_V0)
        
        input_path = args.input
        model_version = args.version
        validate_model = args.validate
        tag = args.tag
        batch_size = args.batch
        net = eval(f'DenseInception_{model_version}()')
        preprocessor = torch.jit.script_if_tracing(Preprocessor())
        device = torch.device("cuda:0")

        # check whether selected model has been downloaded
        model_artifact_path = f'artifacts/DenseInception_{model_version}:v{tag}/{net.__class__.__name__}.pt'
        pretrained = None
        if not exists(model_artifact_path):
            with wandb.init() as run:
                model_artifact = run.use_artifact(f'dwidlee/intel_image_cls/DenseInception_{model_version}:v{tag}')
                model_artifact.download()

        pretrained = torch.load(model_artifact_path)
        net.load_state_dict(pretrained, strict=False)

        if validate_model:
            val_loss, val_acc = validate(net, device)
            print(f'validation : {val_acc} | {val_loss}')

        input_data = None
        results = []
        if isdir(input_path):
            input_dataset = datasets.ImageFolder(input_path, transform=preprocessor)
            input_data = data.DataLoader(input_dataset, batch_size)
        else:
            with Image.open(input_path) as raw_image:
                input_data = [[preprocessor(raw_image).reshape(1, 3, 150, 150), 0]]
        
        results = predict(net, input_data, device)
        results_conv = [KLS[result] for result in results]
        with open('submission.csv', 'w+') as csv_file:
            csv_writer = csv.writer(csv_file)
            csv_writer.writerows([[res] for res in results])
        print(results_conv[0])



if __name__ == '__main__':
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument("--version", type=str, default="V0")
    arg_parser.add_argument("--tag", type=int, default=87)
    arg_parser.add_argument("--input", type=str, default='data/seg_pred')
    arg_parser.add_argument("--batch", type=int, default=16)
    arg_parser.add_argument("--validate", type=bool, default=False)
    args = arg_parser.parse_args()
    main(args)
    

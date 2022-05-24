from email.policy import strict
from genericpath import isdir
from pip import main
import wandb
import torch
from model import KLS
from model import DenseInception_V0
from model import Preprocessor
import argparse
from torchvision import datasets
from tqdm import tqdm
from torch.utils import data
from PIL import Image
from model import Preprocessor

def predict(net, data, device):
    net = net.to(device)
    result = []
    for X, _ in tqdm(data):
        X = X.to(device)
        y_hat = net(X)
        result.extend([int(i) for i in y_hat.argmax(axis=1)])
    return result


def main(args):
    with wandb.init() as run:
        print(DenseInception_V0)
        
        input_path = args.input
        model_version = args.version
        tag = args.tag
        batch_size = args.batch
        net = eval(f'DenseInception_{model_version}()')
        preprocessor = torch.jit.script_if_tracing(Preprocessor())
        device = torch.device("cuda:0")
        
        model_artifact = run.use_artifact(f'dwidlee/intel_image_cls/DenseInception_{model_version}:v{tag}')
        model_path = model_artifact.download()

        pretrained = torch.load(f"{model_path}/{net.__class__.__name__}.pt")
        net.load_state_dict(pretrained, strict=False)

        input_data = None
        if isdir(input_path):
            input_dataset = datasets.ImageFolder(input_path, transform=preprocessor)
            input_data = data.DataLoader(input_dataset, batch_size)
        else:
            with Image.open(input_path) as raw_image:
                input_data.append(raw_image)
        
        results = predict(net, input_data, device)
        print(results)
        results_conv = [KLS[result] for result in results]
        print(results_conv[0])



if __name__ == '__main__':
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument("--version", type=str, default="V0")
    arg_parser.add_argument("--tag", type=int, default=87)
    arg_parser.add_argument("--input", type=str, default='data/seg_pred')
    arg_parser.add_argument("--batch", type=int, default=16)
    args = arg_parser.parse_args()
    main(args)
    

from genericpath import isdir
from pip import main
import wandb
import torch
from model import DenseInception_V0
from model import Preprocessor
import argparse
import os
import sys
from PIL import Image

def predict(net, data, device):
    net = net.to(device)
    result = []
    for X in data:
        X = X.to(device)
        y_hat = net(X)
        result.append(int(y_hat.argmax(axis=1)))
    return result


def main(args):
    with wandb.init() as run:
        
        input_path = args.input
        model_version = args.version
        tag = args.tag
        net = eval(f'DenseInception_V{model_version}()')
        run.use_artifact(f'dwidlee/intel_image_cls/DenseInception_V{model_version}:v{tag}')
        if isdir(input_path):
            pass
        else:
            with Image.open(input_path) as f:
                data = [f]
    



if __name__ == '__main__':
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument("--version", type=str, default="V0")
    arg_parser.add_argument("--tag", type=int, default=87)
    arg_parser.add_argument("--input", type=str, default='data/seg_pred')
    args = arg_parser.parse_args()
    main(args)
    

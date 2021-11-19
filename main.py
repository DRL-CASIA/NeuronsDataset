# import from local source codes
# from src.neuronsdataset.dataset import FrameDataset
# from src.neuronsdataset.visualizer import Visualizer
# from src.neuronsdataset.sentryDataset import sentryDataset
# from src.neuronsdataset.rgbDataset import rgbDataset
# from src.neuronsdataset.rgbdDataset import rgbdDataset
from neuronsdataset.dataset import FrameDataset
from neuronsdataset.visualizer import Visualizer
from neuronsdataset.sentryDataset import sentryDataset
from neuronsdataset.rgbDataset import rgbDataset
from neuronsdataset.rgbdDataset import rgbdDataset

import torchvision.transforms as T
import torch
import argparse
import numpy as np
import os

def train(args):
    if args.seed is not None:
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        torch.backends.cudnn.benchmark = True
    else:
        torch.backends.cudnn.benchmark = True

    dataset = args.dataset
    dataroot = args.root

    if dataset == "sentry":
        base = sentryDataset(os.path.join(dataroot, dataset), worldgrid_shape=[449, 800])
        train_set =  FrameDataset(base, grid_reduce=4, img_reduce=4, train_ratio=args.train_ratio, train=True)
        test_set =  FrameDataset(base, grid_reduce=4, img_reduce=4, train_ratio=args.train_ratio, train=False)
    if dataset == "rgb":
        base = rgbDataset(os.path.join(dataroot, dataset))
        train_set = FrameDataset(base, train_ratio=args.train_ratio, train=True)
        test_set = FrameDataset(base, train_ratio=args.train_ratio, train=False)
    if dataset == "rgbd":
        base = rgbdDataset(os.path.join(dataroot, dataset))
        train_set = FrameDataset(base, train_ratio=args.train_ratio, train=True)
        test_set = FrameDataset(base, train_ratio=args.train_ratio, train=False)

    normalize = T.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))

    train_trans = T.Compose([T.ToTensor(), normalize])
    test_trans = T.Compose([T.ToTensor(), normalize])

    train_loader = torch.utils.data.DataLoader(train_set, batch_size=args.batch_size, shuffle=True,num_workers=args.num_workers, pin_memory=True, drop_last=True)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=args.batch_size, shuffle=True,num_workers=args.num_workers, pin_memory=True, drop_last=True)

    for batch_idx, data in enumerate(train_loader):
        print(data)
        break

def visualize_dataset(args):
    vis = Visualizer(args.root, args.dataset)
    vis.visualize(args.idx)


if __name__ == "__main__":
    # Basic Settings
    parser = argparse.ArgumentParser(description='Dataloader API')
    parser.add_argument('-d', '--dataset', type=str, default='sentry', choices=['sentry', 'rgb', 'rgbd'])
    parser.add_argument('-p', '--root', type=str, default="sample_data")
    parser.add_argument('-i', '--idx', type=int, default=5)

    # Hyper Parameters
    parser.add_argument('-s', '--seed', type=int, default=7)
    parser.add_argument('-t', '--train_ratio', type=int, default=0.8)
    parser.add_argument('-b', '--batch_size', type=int, default=1)
    parser.add_argument('-w', '--num_workers', type=int, default=8)
    args = parser.parse_args()

    # Sample Commands
    train(args)
    visualize_dataset(args)
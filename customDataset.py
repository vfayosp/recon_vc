import numpy as np 
import pandas as pd
import torch
from PIL import Image
from torchvision import datasets,transforms
import glob2
import os

def main():

    transformations = transforms.Compose(
            [transforms.CenterCrop(600),
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(100)]
            )

    train_set = datasets.ImageFolder(root= '/workspace/recon_vc/Database Victor/train',transform= transformations)
    test_set = datasets.ImageFolder(root= '/workspace/recon_vc/Database Victor/test',transform= transformations)



if __name__ == "__main__":
    main()
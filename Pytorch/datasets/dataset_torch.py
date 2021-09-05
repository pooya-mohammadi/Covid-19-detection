"""
This module contains preprocessing and augmentation modules
"""
import torch
from torch.utils.data import DataLoader
from torchvision import transforms, datasets
from torchvision.transforms import transforms
from preparing_datasets import PreparingDatasets


class Dataset:
    @staticmethod
    def pytorch_preprocess(dataset_dir="./covid-19/", img_size=224, batch_size=32, augment=True, split_size=0.3):
        preparing_datasets = PreparingDatasets(framework='pytorch')
        preparing_datasets.preparing_datasets(split_size=split_size)

        train_transforms = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.RandomAffine(5, shear=0.2),
            transforms.RandomHorizontalFlip(),
            transforms.Grayscale(num_output_channels=3),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ])

        test_transforms = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.Grayscale(num_output_channels=3),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])])

        if not augment:
            train_transforms = test_transforms

        train_data = datasets.ImageFolder(dataset_dir + '/train', transform=train_transforms)
        val_data = datasets.ImageFolder(dataset_dir + '/val', transform=test_transforms)
        test_data = datasets.ImageFolder(dataset_dir + '/test', transform=test_transforms)

        train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, shuffle=True)
        val_loader = torch.utils.data.DataLoader(val_data, batch_size=batch_size)
        test_loader = torch.utils.data.DataLoader(test_data, batch_size=batch_size)
        return train_loader, val_loader, test_loader

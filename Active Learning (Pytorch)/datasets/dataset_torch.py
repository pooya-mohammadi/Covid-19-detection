"""
This module contains preprocessing and augmentation modules
"""
import os
import PIL.Image
import filetype
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
from torchvision import transforms
from torchvision.transforms import transforms
from torch.utils.data import Dataset
from datasets.preparing_datasets import PreparingDatasets


class CustomDataset(Dataset):
    def __init__(self, dir_path, transform=None, test=False, split_size=0.3):
        '''
        Args:
        - dir_path (string): path to the directory containing images
        - transform (torchvision.transforms.) (default=None)
        - test (boolean): True for labeled images, False otherwise (default=False)
        '''
        self.dir_path = dir_path
        self.transform = transform
        self.image_filenames = []
        self.labels = []

        preparing_datasets = PreparingDatasets(framework='pytorch')
        preparing_datasets.preparing_datasets(split_size=split_size)

        image_filenames = []
        for (dirpath, dirnames, filenames) in os.walk(dir_path):
            image_filenames += [os.path.join(dirpath, file) for file in filenames if
                                filetype.is_image(os.path.join(dirpath, file))]
            self.image_filenames = image_filenames

        # We assume that in the beginning, the entire dataset is unlabeled, unless it is flagged as 'test':
        if test:
            for f in self.image_filenames:
                if f.split('/')[-2] == 'normal':
                    self.labels.append(0)
                elif f.split('/')[-2] == 'covid':
                    self.labels.append(1)
            self.unlabeled_mask = np.zeros(len(self.image_filenames))
        else:
            self.labels = [0] * len(self.image_filenames)
            self.unlabeled_mask = np.ones(len(self.image_filenames))

    def __len__(self):
        return len(self.image_filenames)

    def __getitem__(self, idx):
        img_name = self.image_filenames[idx]
        img = open(img_name, 'rb')
        image = PIL.Image.open(img)

        if self.transform:
            image = self.transform(image)

        return image, self.labels[idx], idx

    # Display the image [idx] and its filename
    def display(self, idx):
        img_name = self.image_filenames[idx]
        print(img_name)
        img = mpimg.imread(img_name)
        imgplot = plt.imshow(img)
        plt.show()
        return

    # Set the label of image [idx] to 'new_label'
    def update_label(self, idx, new_label):
        self.labels[idx] = new_label
        self.unlabeled_mask[idx] = 0
        return

    # Set the label of image [idx] to that read from its filename
    def label_from_filename(self, idx):
        if self.image_filenames[idx].split('/')[-2] == 'normal':
            self.labels[idx] = 0
        elif self.image_filenames[idx].split('/')[-2] == 'covid':
            self.labels[idx] = 1
        self.unlabeled_mask[idx] = 0
        return

    @staticmethod
    def preparing_datasets(hps):
        train_set = CustomDataset(hps['dataset_dir'] + '/train',
                                  transform=transforms.Compose([
                                      transforms.Resize((hps['img_size'], hps['img_size'])),
                                      transforms.Grayscale(num_output_channels=3),
                                      transforms.ToTensor(),
                                  ]))

        test_set = CustomDataset(hps['dataset_dir'] + '/val',
                                 transform=transforms.Compose([
                                     transforms.Resize((hps['img_size'], hps['img_size'])),
                                     transforms.Grayscale(num_output_channels=3),
                                     transforms.ToTensor()]),
                                 test=True)
        return train_set, test_set

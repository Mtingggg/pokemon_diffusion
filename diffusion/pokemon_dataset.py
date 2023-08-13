import os
import cv2
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

class PokemonDataset(Dataset):
    def __init__(self, imgs_path, transform, data_path=None):
        self.imgs_path = imgs_path
        self.data_path = data_path
        self.imgs_name = os.listdir(imgs_path)
        self.transform = transform
        self.meta_data = None
        self.__read_data()
                
    # Return the number of images in the dataset
    def __len__(self):
        return len(os.listdir(self.imgs_path))
    
    def __read_and_gbr2rgb(self, idx):
        img_path = os.path.join(self.imgs_path, self.imgs_name[idx])
        image = cv2.imread(img_path, cv2.IMREAD_COLOR)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        return image
    
    def __read_data(self):
        if self.data_path:
            pokemon_stats = pd.read_csv(self.data_path)
            pokemon_stats['Image_name'] = pokemon_stats['Image_name']+'.jpg'
            pokemon_type_onehot = pd.get_dummies(pokemon_stats.filter(like='Type').stack()).groupby(level=0).max().astype(int)
            self.meta_data = pd.concat([pokemon_stats[['Image_name']], pokemon_type_onehot], axis=1)
    
    # Get the image and label at a given index
    def __getitem__(self, idx):
        image = self.__read_and_gbr2rgb(idx)
        if self.transform:
            image = self.transform(image)
        if self.meta_data is not None:
            name = self.imgs_name[idx]
            data = torch.tensor(np.array(self.meta_data.query('Image_name==@name').drop(columns='Image_name')), dtype=torch.float32)
            return image, data
        else:
            return image

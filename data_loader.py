import os
from PIL import Image

from torch.utils.data import Dataset


class KittiLoader(Dataset):
    def __init__(self, root_dir, mode, transform=None):
        left_dir = os.path.join(root_dir, 'image_02/data/')
        self.left_paths = sorted([os.path.join(left_dir, fname) for fname\
                           in os.listdir(left_dir)])
                           
        disp_model_based_dir = os.path.join(root_dir, 'disp_model_based/data/')
        self.disp_model_based_paths = sorted([os.path.join(disp_model_based_dir, fname) for fname\
                           in os.listdir(disp_model_based_dir)])
                           
        assert len(self.disp_model_based_paths) == len(self.left_paths)
        
        if mode == 'train':
            right_dir = os.path.join(root_dir, 'image_03/data/')
            self.right_paths = sorted([os.path.join(right_dir, fname) for fname\
                                in os.listdir(right_dir)])
            assert len(self.right_paths) == len(self.left_paths)
            
        self.transform = transform
        self.mode = mode


    def __len__(self):
        return len(self.left_paths)

    def __getitem__(self, idx):
        left_image = Image.open(self.left_paths[idx])
        disp_model_based_image = Image.open(self.disp_model_based_paths[idx])
        
        if self.mode == 'train':
            right_image = Image.open(self.right_paths[idx])
            sample = {'left_image': left_image, 'right_image': right_image, 'disp_model_based' : disp_model_based}
        else:
            sample = {'left_image': left_image, 'disp_model_based' : disp_model_based}
            
        if self.transform:
            sample = self.transform(sample)
            return sample
        else:
            return sample

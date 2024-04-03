import os
from PIL import Image
from torch.utils.data import Dataset
import numpy as np

class PeopleDataset(Dataset):
    def __init__(self, image_dir, mask_dir, transform=None):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.transform = transform
        self.image = os.listdir(image_dir)
    
    def __len__(self):
        return len(self.image)
    
    def __getItem__(self,index):
        img_path = os.path.join(self.image_dir, self.image[index])
        mask_path = os.path.join(self.mask_dir, self.image[index])
        image = np.array(Image.open(img_path).convert("RGB"))
        mask = np.array(Image.open(mask_path).convert("L"), dtype=np.float32)
        mask[mask == 255.0] = 1.0

        if self.transform is not None:
            augmentation = self.transform(image = image, mask= mask)
            image = augmentation['image']
            mask = augmentation['mask']
            
        return image, mask


    

    


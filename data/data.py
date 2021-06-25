import os
import glob

import torchvision.transforms as transforms

from torch.utils.data import Dataset
from PIL import Image



class DATA(Dataset):
    def __init__(self, args, mode='train'):

        ''' set up basic parameters for dataset '''
        self.train_dir = args.train_dir
        self.test_dir = args.test_dir
        if mode == 'train':
            self.img_dir = os.path.join(self.train_dir, 'pristine_images')
            self.file_names = sorted(glob.glob(self.img_dir))
        if mode == 'test':
            self.file_names = sorted(glob.glob(self.test_dir))

        self.transform = transforms.Compose([
            transforms.RandomCrop(192),
            transforms.ToTensor(),  # (H,W,C)->(C,H,W), [0,255]->[0, 1.0] RGB->RGB
        ])

    def __len__(self):
        return len(self.file_names)

    def __getitem__(self, idx):

        ''' get data '''
        img_path = self.file_names[idx]

        ''' read image '''
        img = Image.open(img_path)

        return self.transform(img)
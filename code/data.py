import os
import glob

import torchvision.transforms as transforms

from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from PIL import Image
import time
'''
transform = transforms.Compose([
            transforms.RandomCrop(192),
            transforms.ToTensor(),  # (H,W,C)->(C,H,W), [0,255]->[0, 1.0] RGB->RGB
        ])
a= time.time()
img = transform(Image.open('train_data_waterloo/pristine_images/00001.bmp'))
print((time.time()-a)*16)
'''
class DATA(Dataset):
    def __init__(self, args, mode='train'):

        ''' set up basic parameters for dataset '''
        self.train_dir = args.train_dir
        self.test_dir = args.test_dir
        if mode == 'train':
            self.img_dir = os.path.join(self.train_dir, 'pristine_images')
            self.file_names = sorted(glob.glob(os.path.join(self.img_dir, '*.bmp')))
        if mode == 'test':
            self.file_names = sorted(glob.glob(os.path.join(self.test_dir, '*.png')))

        self.transform = transforms.Compose([
            transforms.RandomCrop(192),
            transforms.ToTensor(),  # (H,W,C)->(C,H,W), [0,255]->[0, 1.0] RGB->RGB
        ])

        self.tensors = list()
        for file_name in self.file_names:
            img = Image.open(file_name)
            if (self.transform(img).shape[0] != 3):
                print(file_name, self.transform(img).shape)
            if (self.transform(img).shape[0] == 3):
                self.tensors.append(self.transform(img))

    def __len__(self):
        return len(self.tensors)

    def __getitem__(self, idx):

        ''' get data '''
        tensor = self.tensors[idx]
        return tensor

class MultiEpochsDataLoader(DataLoader):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._DataLoader__initialized = False
        self.batch_sampler = _RepeatSampler(self.batch_sampler)
        self._DataLoader__initialized = True
        self.iterator = super().__iter__()

    def __len__(self):
        return len(self.batch_sampler.sampler)

    def __iter__(self):
        for i in range(len(self)):
            yield next(self.iterator)


class _RepeatSampler(object):
    """ Sampler that repeats forever.
    Args:
        sampler (Sampler)
    """

    def __init__(self, sampler):
        self.sampler = sampler

    def __iter__(self):
        while True:
            yield from iter(self.sampler)
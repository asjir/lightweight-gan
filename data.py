import re
from pathlib import Path
from random import random

import torch
from more_itertools import flatten
from PIL.Image import BILINEAR
from torch.functional import norm
from torch.utils.data import Dataset
from torchvision import transforms

from util import dose2locs, identity, loc2dose


def transforms1(image_size, w=3, zoom=1.1, erase_p=0):
    return [
        transforms.Resize(image_size),
        transforms.RandomAffine(w, (.01*w, .01*w), (1, 1), w, BILINEAR),
        transforms.Resize(int(image_size*zoom)), 
        transforms.CenterCrop(image_size)  , 
        transforms.RandomErasing(p=erase_p, scale=(0.02, 0.33), ratio=(0.3, 3.3), value=0)
    ]
    


class DoseCurveDataset(Dataset):
    def __init__(self, folder, image_size, chans=[0,1,2,3,4], train=True, norm_f=None,
                 w=None, doses="all", label=False, multiplier=None, erase_p=0):

        if doses == "all":
            doses = dose2locs.keys()
        w = w or (3 if train else 0)
        
        def paths(folder, doses):
            not_52 = re.compile('/[^(52)]')
            assays = flatten(dose2locs[dose] for dose in doses)
            gen = flatten((Path(f'{folder}').glob(
                f'**/*{assay}*.pt')) for assay in assays)
            return [p for p in gen if not_52.search(str(p))]

        self.dose2id = {k: i for i, k in enumerate(sorted(doses))}
        self.f = d8 if train else identity()
        super().__init__()
        self.folder = folder
        self.image_size = image_size
        self.label = label
        
        assert not norm_f or not multiplier
        if not norm_f and not multiplier: multiplier = 1
        self.norm_f = norm_f or (lambda x: (x*multiplier/255).clamp(0, 1))

        self.paths = paths(folder, doses)
        assert len(self.paths) > 0, f'No images were found in {folder} for training'

        #convert_image_fn = convert_transparent_to_rgb if not transparent else convert_rgb_to_transparent
        self.chans = chans 
        self.transform = transforms.Compose(transforms1(image_size, w, erase_p=erase_p))
        

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, index):
        path = self.paths[index]
        img = self.norm_f(torch.load(path))
        
        img = img[self.chans]
        
        if self.label:
            label = self.dose2id[loc2dose[str(path).split()[-2]]]
            return self.transform(self.f(img)), label
        return (self.transform(self.f(img)))


class MSNorm:  
    def __init__(self, norm_path):
        self.mean, self.std = torch.load(norm_path, map_location='cpu')
        
    def __call__(self, img):
        return (img - self.mean) / self.std

    def invert(self, img):
        return img * self.std + self.mean

    
def denorm_f(ms, device):
    mean, std = map(lambda x: torch.tensor(x,  device=device)[None, :, None, None], ms)
    return lambda x: (x*std + mean).cpu()

def d8(img):
    img = torch.rot90(img, int(random()*4), dims=(1,2))
    if random()>.5:
        img = torch.flip(img, dims=(2,))
    return img

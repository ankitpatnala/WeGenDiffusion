import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import xarray as xr
from torchvision import transforms

class MyDataset(Dataset):
    def __init__(self, data, target, transform=None):
        self.data = torch.from_numpy(data).float()
        self.target = torch.from_numpy(target).long()
        self.transform = transform
        
    def __getitem__(self, index):
        x = self.data[index]
        y = self.target[index]
        
        if self.transform:
            x = self.transform(x)
        
        return x, y
    
    def __len__(self):
        return len(self.data)

ds_train = xr.open_dataset("/fast/project/HFMI_HClimRep/nishant.kumar/dit_hackathon/data/2011_t2m_era5_2deg.nc") 
ds_val   = xr.open_dataset("/fast/project/HFMI_HClimRep/nishant.kumar/dit_hackathon/data/2012_t2m_era5_4months_2deg.nc")

train_data = ds_train['t2m'].values
val_data = ds_val['t2m'].values
train_target = np.zeros((1460,), dtype=int)
val_target  = np.zeros((492,), dtype=int)

# Define the transformations
data_transforms = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

train_dataset = MyDataset(train_data, train_target, transform=data_transforms)
train_loader  = DataLoader(
    train_dataset,
    batch_size=10,
    shuffle=True,
    num_workers=2,
    pin_memory=torch.cuda.is_available()
)

val_dataset = MyDataset(val_data, val_target, transform=data_transforms)
val_loader  = DataLoader(
    val_dataset,
    batch_size=10,
    shuffle=True,
    num_workers=2,
    pin_memory=torch.cuda.is_available()
)
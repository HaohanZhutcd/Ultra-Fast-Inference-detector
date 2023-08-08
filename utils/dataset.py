import torch, os
from torch import nn
from torch.utils.data import DataLoader
from torchvision.transforms import transforms
from torchvision.datasets import ImageFolder
import pytorch_lightning as pl
from torch.utils.data import Dataset
from PIL import Image
import torch.nn.functional as F
import argparse
import os
import platform
import sys
from pathlib import Path

# class cropDataset(Dataset):
#     def __init__(self, folder, transform, time_step):
#         self.folder = folder
#         self.transform = transform
#         self.time_step = time_step
#         cls_dict = {}
#         # print(self.folder)
#         # folder = 'crop_imgs'
#         self.cls_dict, self.cls_name = [], []
#         folder_list = [i for i in os.listdir(self.folder) if i[0] != '.']
#         # print(folder_list)
#         for i in folder_list:
#             cls_file_name_list = [j for j in os.listdir(os.path.join(self.folder, i)) if j[0] != '.']
#             cls_file_name_list.sort()
#             # print(cls_file_name_list)
#             prefix_set = set([name.split('_')[0]+'_'+ name.split('_')[1] for name in cls_file_name_list ])
#
#             # if len(prefix_set)!=0:
#             #     # cls_dict[i]=[]
#             for j in prefix_set:
#                 cls_prefix_list = [os.path.join(self.folder, i, name) for name in cls_file_name_list if j in name]
#                 cls_prefix_list.sort()
#                 for num in range(len(cls_prefix_list) // self.time_step ):
#                     self.cls_dict.append(cls_prefix_list[self.time_step  * num:self.time_step  * num + self.time_step ])
#                     self.cls_name.append(i)
#         # filenames_list=[]
#         # for filepath,dirnames,filenames in os.walk(folder):
#         #         for filename in filenames:
#         #             filenames_list.append(filename)
#         self.cls_index = list(set(self.cls_name))
#         # print(self.cls_index)
#
#     def __getitem__(self, idx):
#         tensor_list = []
#         for i in self.cls_dict[idx]:
#             img = Image.open(i)
#             img_tensor = self.transform(img)
#             tensor_list.append(img_tensor)
#         img_tensor = torch.stack(tensor_list)
#         # print(img_tensor.size())
#         cls = self.cls_index.index(self.cls_name[idx])
#         return img_tensor, cls
#
#     def __len__(self):
#
#         return len(self.cls_name)

class cropDataset(Dataset):
    def __init__(self, folder, transform, time_step):
        self.folder = folder
        self.transform = transform
        self.time_step = time_step
        cls_dict = {}
        self.cls_dict, self.cls_name = [], []
        folder_list = [i for i in os.listdir(self.folder) if i[0] != '.']
        for i in folder_list:
            cls_file_name_list = [j for j in os.listdir(os.path.join(self.folder, i)) if j[0] != '.']
            cls_file_name_list.sort()
            prefix_set = set([name.split('_')[0] + '_' + name.split('_')[1] for name in cls_file_name_list])
            for j in prefix_set:
                cls_prefix_list = [os.path.join(self.folder, i, name) for name in cls_file_name_list if j in name]
                cls_prefix_list.sort()
                self.cls_dict.append(cls_prefix_list)
                self.cls_name.append(i)

        self.cls_index = list(set(self.cls_name))
        print(self.cls_index)

    def __getitem__(self, idx):
        tensor_list = []
        cls_prefix_list = self.cls_dict[idx]
        if len(cls_prefix_list) >= self.time_step:
            start_idx = torch.randint(0, len(cls_prefix_list) - self.time_step + 1, (1,)).item()
            cls_prefix_list = cls_prefix_list[start_idx : start_idx + self.time_step]
        else:
            cls_prefix_list = cls_prefix_list + [cls_prefix_list[-1]] * (self.time_step - len(cls_prefix_list))

        for i in cls_prefix_list:
            img = Image.open(i)
            img_tensor = self.transform(img)
            tensor_list.append(img_tensor)
        img_tensor = torch.stack(tensor_list)

        cls = self.cls_index.index(self.cls_name[idx])
        return img_tensor, cls

    def __len__(self):
        return len(self.cls_name)


if __name__ == '__main__':
    FILE = Path(__file__).resolve()
    ROOT = FILE.parents[0]  # YOLOv5 root directory
    if str(ROOT) not in sys.path:
        sys.path.append(str(ROOT))  # add ROOT to PATH
    ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    train_dataset = cropDataset('../data/crop_imgs', transform=transform, time_step=8)
    val_dataset = cropDataset('../data/crop_imgs_val', transform=transform, time_step=8)

    train_dataloader = DataLoader(train_dataset, batch_size=1, shuffle=False)

    for batch_idx, batch_data in enumerate(train_dataloader):
        # 'batch_data' is a tuple containing the input data and target labels
        # If your DataLoader is designed to return (inputs, targets) for each batch,
        # then you can unpack the tuple like this:
        inputs, targets = batch_data

        # Now you can print the batch data
        print(f"Batch {batch_idx + 1}:")
        print("Input data shape:", inputs.shape)
        print("Target labels shape:", targets)
        print("-----------------------")

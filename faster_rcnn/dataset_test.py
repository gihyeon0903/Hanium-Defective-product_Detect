import os, cv2, json, random, math
import numpy as np
from numpy.random import RandomState
from skimage.measure import regionprops

import torch
from torch import Tensor
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms

import imgaug as ia
from imgaug import augmenters as iaa


class custom_dataset(Dataset):
    
    def __init__(self, device):
        self.device = device
        self.to_tensor = transforms.ToTensor()
        self.flip = iaa.Fliplr(0.5)
        self.resize = iaa.Resize({"shorter-side": 600, "longer-side": "keep-aspect-ratio"})
        
        self.img_folder_root = "./test_data/img/"
        
        # json load
        json_root1 = "./test_data/json_file/annotation1.json"
        json_root2 = "./test_data/json_file/annotation2.json"
        
        with open(json_root1, 'r', encoding='UTF8') as f:
            json_data = json.load(f)
            self.json_data = json_data
        
#         with open(json_root2, 'r', encoding='UTF8') as f:
#             json_data = json.load(f)
#             self.json_data.update(json_data)

        file_list = np.fromiter(self.json_data.keys(), dtype = 'U64')
        
        self.file_name = []
        self.bbox = []
        

        for file_list_ in file_list:
            img_name_list = self.json_data[file_list_]['filename']
            self.file_name.append(img_name_list)
            
            objs = self.json_data[file_list_]['regions']

            bbox = []
            for obj in objs:
                obj_info = obj['shape_attributes']

                x, y, w, h = obj_info['x'], obj_info['y'], obj_info['width'], obj_info['height']
                one_bbox = [x, y, x+w, y+h]

                bbox.append(one_bbox)
                
            self.bbox.append([bbox])
        
    def __len__(self):
        return len(self.file_name)
        
    def __getitem__(self, idx):
        
        ff = np.fromfile(self.img_folder_root + self.file_name[idx], np.int8)
        cv_image = cv2.imdecode(ff, cv2.IMREAD_UNCHANGED)
        cv_image = cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB)
        
        bbox = np.array(self.bbox[idx])
        
        cv_image, bbox = self.flip(image = cv_image, bounding_boxes = bbox)
        cv_image, bbox = self.resize(image = cv_image, bounding_boxes = bbox)
        
        bbox = bbox.squeeze(0).tolist()
        image = self.to_tensor(cv_image)

        targets = []
        d = {}
        d['boxes'] = torch.tensor(bbox, device=self.device)
        d['labels'] = torch.tensor([1 for x in range(len(bbox))],dtype=torch.int64, device=self.device)
        targets.append(d)

        return image, targets

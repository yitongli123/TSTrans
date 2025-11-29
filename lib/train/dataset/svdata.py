import os
import os.path
import numpy as np
import torch
import csv
import pandas
import random
from collections import OrderedDict
from .base_video_dataset import BaseVideoDataset
from lib.train.data import jpeg4py_loader
from lib.train.admin import env_settings


class SVdata(BaseVideoDataset):
    """ SVdata dataset.
    """

    def __init__(self, root=None, image_loader=jpeg4py_loader, split=None, data_fraction=None):
        """
        args:
            root - path to the training and validation data.
            image_loader (jpeg4py_loader) -  The function to read the images. jpeg4py (https://github.com/ajkxyz/jpeg4py)
                                            is used by default.
            split - specific dataset name + '-' + 'train' or 'val'. Note: The validation split here is the same as the testing split only for
                    showing the trained model's performance after different epochs during model training.
            data_fraction - Fraction of dataset to be used. The complete dataset is used by default
        """
        root = env_settings().svdata_dir if root is None else root  # root需要在环境变量文件lib/train/admin/local.py中设置'svdata_dir'为数据存储的文件夹路径
        super().__init__('SVdata', root, image_loader)

        assert split is not None
        ltr_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), '..')
        if split == 'sv248s-train':  
            file_path = os.path.join(ltr_path, 'data_specs', 'sv248s/train_split.txt')               
        elif split == 'sv248s-val':  
            file_path = os.path.join(ltr_path, 'data_specs', 'sv248s/val_split.txt')
        elif split == 'satsot-train':  
            file_path = os.path.join(ltr_path, 'data_specs', 'satsot/train_split.txt')               
        elif split == 'satsot-val':  
            file_path = os.path.join(ltr_path, 'data_specs', 'satsot/val_split.txt')
        elif split == 'viso-train':  
            file_path = os.path.join(ltr_path, 'data_specs', 'viso/train_split.txt')               
        elif split == 'viso-val':  
            file_path = os.path.join(ltr_path, 'data_specs', 'viso/val_split.txt')
        else:
            raise ValueError('Unknown split name.')
        with open(file_path, 'r') as f:
            self.sequence_list = f.readlines()
        self.sequence_list = [line.strip() for line in self.sequence_list]

        if data_fraction is not None:
            # not into this branch
            self.sequence_list = random.sample(self.sequence_list, int(len(self.sequence_list)*data_fraction))

    def get_name(self):
        return 'SVdata'

    def has_class_info(self):
        return True

    def has_occlusion_info(self):
        return True

    def _read_bb_anno(self, seq_path):
        #print(seq_path) -> root_path + seq_name
        seq_name = seq_path.split('/')[-1]
        if 'car' in seq_name or 'plane' in seq_name or 'ship' in seq_name or 'train' in seq_name:  # SatSOT
            bb_anno_file = os.path.join(seq_path, "groundtruth.txt")
        elif 'vehicle' in seq_name or 'aero' in seq_name or 'boat' in seq_name or 'rail' in seq_name:  # VISO-SOT
            bb_anno_file = os.path.join(seq_path, seq_path.split('/')[-1]+'.txt')
        else:  # SV248S
            bb_anno_file = os.path.join(seq_path, seq_path.split('/')[-1]+'.rect')
        gt = pandas.read_csv(bb_anno_file, delimiter=',', header=None, dtype=np.float32, na_filter=False, low_memory=False).values
        #print(gt)
        return torch.tensor(gt)

    def _read_target_visible(self, seq_path):
        seq_name = seq_path.split('/')[-1]
        cls_list = ['car', 'vehicle', 'plane', 'aero', 'ship', 'boat', 'train', 'rail']
        if any(True if cls in seq_name else False for cls in cls_list):  # SatSOT & VISO-SOT
            if 'car' in seq_name or 'plane' in seq_name or 'ship' in seq_name or 'train' in seq_name:  # SatSOT
                bb_anno_file = os.path.join(seq_path, "groundtruth.txt") 
            else:  # VISO-SOT
                bb_anno_file = os.path.join(seq_path, seq_path.split('/')[-1]+'.txt') 
            occlusion = []
            cover = []
            with open(bb_anno_file, 'r') as f:  # there isn't frame attribute, thus all-visible
                gt = f.readlines()
                for v in gt:
                    if v != '\n':
                        if '0,0,0,0' in v:
                            occlusion.append(int(1))
                            cover.append(int(0))
                        else:
                            occlusion.append(int(0))
                            cover.append(int(8))
                occlusion = torch.ByteTensor(occlusion)
                cover = torch.ByteTensor(cover)

            target_visible = ~occlusion & (cover>0).byte()  # !!!取反occlusion，和cover做与运算

            visible_ratio = cover.float() / 8
        else:  # SV248S 
            # Read full occlusion and out_of_view
            occlusion_file = os.path.join(seq_path, seq_path.split('/')[-1]+'.state')
            cover_file = os.path.join(seq_path, seq_path.split('/')[-1]+'.state')

            with open(occlusion_file, 'r', newline='') as f:
                # occlusion = torch.ByteTensor([int(v[0]) for v in csv.reader(f)])
                occlusion_list = []
                for v in csv.reader(f):
                    if int(v[0]) == 0:  # 帧属性为NOR->0(Normal Visiable)
                        occlusion_list.append(int(0))  # The object is present
                    else:  # 帧属性为INV->1(Invisiable)或者OCC->2(Occlusion)
                        occlusion_list.append(int(1))  # The object is absent
                occlusion = torch.ByteTensor(occlusion_list)  
            with open(cover_file, 'r', newline='') as f:
                # cover = torch.ByteTensor([int(v[0]) for v in csv.reader(f)])
                cover_list = []
                for v in csv.reader(f):
                    if int(v[0]) == 0:  # 帧属性为NOR->0(Normal Visiable)
                        cover_list.append(int(8))
                    else:  # 帧属性为INV->1(Invisiable)或者OCC->2(Occlusion)
                        cover_list.append(int(0))
                cover = torch.ByteTensor(cover_list)

            target_visible = ~occlusion & (cover>0).byte()  # !!!取反occlusion，和cover做与运算

            visible_ratio = cover.float() / 8
        return target_visible, visible_ratio

    def _get_sequence_path(self, seq_id):
        return os.path.join(self.root, self.sequence_list[seq_id])

    def get_sequence_info(self, seq_id):
        seq_path = self._get_sequence_path(seq_id)
        bbox = self._read_bb_anno(seq_path)

        valid = (bbox[:, 2] > 0) & (bbox[:, 3] > 0)
        visible, visible_ratio = self._read_target_visible(seq_path)
        visible = visible & valid.byte()

        return {'bbox': bbox, 'valid': valid, 'visible': visible, 'visible_ratio': visible_ratio}

    def _get_frame_path(self, seq_path, frame_id):
        seq_name = seq_path.split('/')[-1]
        if 'car' in seq_name or 'plane' in seq_name or 'ship' in seq_name or 'train' in seq_name:  # SatSOT
            seq_path = os.path.join(seq_path, 'img')
            if os.path.exists(os.path.join(seq_path, '0001.jpg')):
                frame_path = os.path.join(seq_path, '{:04}.jpg'.format(frame_id+1))
            else:
                files = [i for i in os.listdir(seq_path) if 'jpg' in i]
                files.sort(key=lambda x: int(x.split('.')[0]))
                start_id = int(files[0].split('.')[0])
                frame_path = os.path.join(seq_path, '{:04}.jpg'.format(frame_id+start_id))
        elif 'vehicle' in seq_name or 'aero' in seq_name or 'boat' in seq_name or 'rail' in seq_name:  # VISO-SOT
            if os.path.exists(os.path.join(seq_path, '000001.jpg')):
                frame_path = os.path.join(seq_path, '{:06}.jpg'.format(frame_id+1))
            else:
                files = [i for i in os.listdir(seq_path) if 'jpg' in i]
                files.sort(key=lambda x: int(x.split('.')[0]))
                start_id = int(files[0].split('.')[0])
                frame_path = os.path.join(seq_path, '{:06}.jpg'.format(frame_id+start_id))
        else:  # SV248S
            #print(os.path.join(seq_path, '{:08}.jpg'.format(frame_id+1))) # frames start from 1
            if os.path.exists(os.path.join(seq_path, '000001.tiff')):
                frame_path = os.path.join(seq_path, '{:06}.tiff'.format(frame_id+1))
            else:
                files = [i for i in os.listdir(seq_path) if 'tiff' in i]
                files.sort(key=lambda x: int(x.split('.')[0]))
                start_id = int(files[0].split('.')[0])
                #print('start_id of', seq_path.split('/')[-1], 'is', files[0].split('.')[0])
                #with open(os.path.join(seq_path, seq_path.split('/')[-1]+'.abs'),'r') as abs_f:
                    #dict = abs_f.readlines()[0]
                    #dict = json.loads(dict)   
                    #start_id = int(dict["source_info"]["frame_range"][0])             
                frame_path = os.path.join(seq_path, '{:06}.tiff'.format(frame_id+start_id))
        # 已加上起始帧id 
        return  frame_path    

    def _get_frame(self, seq_path, frame_id):
        return self.image_loader(self._get_frame_path(seq_path, frame_id))

    def get_frames(self, seq_id, frame_ids, anno=None):
        seq_path = self._get_sequence_path(seq_id)

        frame_list = [self._get_frame(seq_path, f_id) for f_id in frame_ids]

        if anno is None:
            anno = self.get_sequence_info(seq_id)

        anno_frames = {}
        for key, value in anno.items():
            anno_frames[key] = [value[f_id, ...].clone() for f_id in frame_ids]

        return frame_list, anno_frames

    def get_annos(self, seq_id, frame_ids, anno=None):
        if anno is None:
            anno = self.get_sequence_info(seq_id)

        anno_frames = {}
        for key, value in anno.items():
            anno_frames[key] = [value[f_id, ...].clone() for f_id in frame_ids]

        return anno_frames

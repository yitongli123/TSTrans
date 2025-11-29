import numpy as np
from lib.test.evaluation.data import Sequence, BaseDataset, SequenceList
from lib.test.utils.load_text import load_text
import os


class SVDataset(BaseDataset):
    def __init__(self, split):
        super().__init__()
        self.base_path = self.env_settings.svdataset_path  # 需要在环境变量文件lib/test/evaluation/local.py中设置'svdataset_path'为数据存储的总文件夹路径

        self.sequence_list = self._get_sequence_list(split)
        self.split = split

    def get_sequence_list(self):
        return SequenceList([self._construct_sequence(s) for s in self.sequence_list])

    def _construct_sequence(self, sequence_name):  # 测试时读取视频中的所有帧，不会根据gt筛去invisible
        if 'car' in sequence_name or 'plane' in sequence_name or 'ship' in sequence_name or 'train' in sequence_name:
            anno_path = '{}/{}/groundtruth.txt'.format(self.base_path, sequence_name)

            ground_truth_rect = load_text(str(anno_path), delimiter=',', dtype=np.float64)

            frames_path = '{}/{}/img'.format(self.base_path, sequence_name)
            frame_list = [frame for frame in os.listdir(frames_path) if frame.endswith(".jpg")]
            frame_list.sort(key=lambda f: int(f[:-4]))
            frames_list = [os.path.join(frames_path, frame) for frame in frame_list]
            #print(sequence_name,':\n',frames_list)
        elif 'vehicle' in sequence_name or 'aero' in sequence_name or 'boat' in sequence_name or 'rail' in sequence_name:  # VISO-SOT
            anno_path = '{}/{}/{}.txt'.format(self.base_path, sequence_name, sequence_name)

            ground_truth_rect = load_text(str(anno_path), delimiter=',', dtype=np.float64)

            frames_path = '{}/{}'.format(self.base_path, sequence_name)  
            frame_list = [frame for frame in os.listdir(frames_path) if frame.endswith(".jpg")] 
            frame_list.sort(key=lambda f: int(f[:-4]))  
            frames_list = [os.path.join(frames_path, frame) for frame in frame_list]
        else:
            anno_path = '{}/{}/{}.rect'.format(self.base_path, sequence_name, sequence_name)

            ground_truth_rect = load_text(str(anno_path), delimiter=',', dtype=np.float64)

            frames_path = '{}/{}'.format(self.base_path, sequence_name)
            frame_list = [frame for frame in os.listdir(frames_path) if frame.endswith(".tiff")]
            frame_list.sort(key=lambda f: int(f[:-5]))
            frames_list = [os.path.join(frames_path, frame) for frame in frame_list]
            #print(sequence_name,':\n',frames_list)

        return Sequence(sequence_name, frames_list, 'svdata', ground_truth_rect.reshape(-1, 4))

    def __len__(self):
        return len(self.sequence_list)

    def _get_sequence_list(self, split):
        # 测试时读取的sequence文件路径在lib/test/evaluation/local.py中设置
        assert split in ['sv248s-test', 'satsot-test', 'viso-test']
        if split == 'sv248s-test':
            split_path = self.env_settings.sv248s_path
        if split == 'satsot-test':
            split_path = self.env_settings.satsot_path
        if split == 'viso-test':
            split_path = self.env_settings.viso_path
        with open(split_path,'r') as f:
            sequence_list = f.read().splitlines()

        return sequence_list

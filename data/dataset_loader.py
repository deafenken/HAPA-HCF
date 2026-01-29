import torch.utils.data as data
import numpy as np
import util.utils as utils
import os

class Dataset(data.Dataset):

    def __init__(self, args, transform=None, test_mode=False, return_name=False):
        if test_mode:
            self.rgb_list_file = args.test_rgb_list
            self.audio_list_file = args.test_audio_list
            self.flow_list_file = args.test_flow_list
        else:
            self.rgb_list_file = args.rgb_list
            self.audio_list_file = args.audio_list
            self.flow_list_file = args.flow_list
        self.max_seqlen = args.max_seqlen
        self.transform = transform
        self.test_mode = test_mode
        self.return_name = return_name
        self.normal_flag = '_label_A'
        self.root_dir = 'F:\\test\\code2\\MAVD\\XDData'
        self._parse_list()

    def _parse_list(self):
        self.list = list(open(self.rgb_list_file, encoding='utf-8'))
        self.audio_list = list(open(self.audio_list_file, encoding='utf-8'))
        self.flow_list = list(open(self.flow_list_file, encoding='utf-8'))

    def _fix_path(self, original_path):
        original_path = original_path.strip('\n')
        parts = original_path.split('/')
        file_name = parts[-1]
        sub_folder = parts[-2]
        new_path = os.path.join(self.root_dir, sub_folder, file_name)
        return new_path

    def __getitem__(self, index):
        if self.normal_flag in self.list[index]:
            label = 0.0
        else:
            label = 1.0
        path_v = self._fix_path(self.list[index])
        path_f = self._fix_path(self.flow_list[index])
        path_a = self._fix_path(self.audio_list[index // 5])
        try:
            f_v = np.array(np.load(path_v), dtype=np.float32)
            f_f = np.array(np.load(path_f), dtype=np.float32)
            f_a = np.array(np.load(path_a), dtype=np.float32)
        except FileNotFoundError as e:
            print(f'Error loading file: {e}')
            print(f'Attempted path: {path_v} (or flow/audio equivalent)')
            raise e
        if self.transform is not None:
            f_v = self.transform(f_v)
            f_f = self.transform(f_f)
            f_a = self.transform(f_a)
        if self.test_mode:
            if self.return_name == True:
                file_name = self.list[index].strip('\n').split('/')[-1][:-7]
                return (f_v, f_a, f_f, file_name)
            return (f_v, f_a, f_f)
        else:
            f_v = utils.process_feat(f_v, self.max_seqlen, is_random=False)
            f_a = utils.process_feat(f_a, self.max_seqlen, is_random=False)
            f_f = utils.process_feat(f_f, self.max_seqlen, is_random=False)
            if self.return_name == True:
                file_name = self.list[index].strip('\n').split('/')[-1][:-7]
                return (f_v, f_a, f_f, file_name)
            return (f_v, f_a, f_f, label)

    def __len__(self):
        return len(self.list)
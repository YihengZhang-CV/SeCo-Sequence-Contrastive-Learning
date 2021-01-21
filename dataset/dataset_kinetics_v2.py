import torch.utils.data
import os
import random
import torch
import numpy as np
from PIL import Image
from .dataset_builder import DATASETS


@DATASETS.register_module()
class KineticsClipFolderDatasetV2(torch.utils.data.Dataset):
    def __init__(self, root, split='train_list', **kwargs):
        super(KineticsClipFolderDatasetV2, self).__init__()
        if '##' in root:  # super resource
            data_root_split = root.split('##')
            assert len(data_root_split) == 2
            root = data_root_split[0]
            self.dataset_frame_root_ssd = os.path.join(data_root_split[1], 'data')
            assert '#' not in self.dataset_frame_root_ssd
            assert os.path.exists(self.dataset_frame_root_ssd)
        else:
            self.dataset_frame_root_ssd = None
        # dataset root
        if '#' in root:  # multiple data resources
            self.dataset_root = root.split('#')
        else:
            self.dataset_root = [root]
        for p in self.dataset_root:
            if not os.path.exists(p):
                print(p)
                assert False
        self.dataset_root_num = len(self.dataset_root)
        print('using {} data sources'.format(self.dataset_root_num))
        # data frame root
        self.dataset_frame_root = [os.path.join(p, 'data') for p in self.dataset_root]
        for p in self.dataset_frame_root:
            assert os.path.exists(p)
        # data list file
        assert split in ('train_list', 'val_list')
        self.dataset_list_file = os.path.join(self.dataset_root[0], split + '.txt')
        assert os.path.exists(self.dataset_list_file)
        # load vid samples
        self.samples = self._load_list(self.dataset_list_file)
        self.transform = None

    def _get_aug_frame(self, frame_root, frame_idx):
        frame = Image.open(os.path.join(frame_root, 'frame_{:06d}.jpg'.format(frame_idx)))
        frame.convert('RGB')
        if self.transform is not None:
            frame_aug = self.transform(frame)
        else:
            frame_aug = frame
        return frame_aug

    def _load_list(self, list_root):
        with open(list_root, 'r') as f:
            lines = f.readlines()
        vids = []
        for k, l in enumerate(lines):
            lsp = l.strip().split(' ')
            # path, frame, label
            if self.dataset_frame_root_ssd is not None and os.path.exists(os.path.join(self.dataset_frame_root_ssd, lsp[0])):
                vid_root = os.path.join(self.dataset_frame_root_ssd, lsp[0])
            else:
                vid_root = os.path.join(self.dataset_frame_root[k % self.dataset_root_num], lsp[0])
            vids.append((vid_root, int(lsp[1]), int(lsp[2])))
        return vids

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, item):
        raise NotImplementedError


@DATASETS.register_module()
class KineticsClipFolderDatasetV2MultiFrames(KineticsClipFolderDatasetV2):
    def __init__(self, root, transform=None, split='train_list', sample_num=0):
        super(KineticsClipFolderDatasetV2MultiFrames, self).__init__(root, split)
        self.transform = transform
        self.sample_num = sample_num
        assert self.transform is not None

    def __getitem__(self, item):
        frame_root, frame_num, cls = self.samples[item]
        sample_num = frame_num if self.sample_num <= 0 or self.sample_num > frame_num else self.sample_num
        frame_indices = np.round(np.linspace(1, frame_num, num=sample_num)).astype(np.int64)
        frames = torch.cat([self._get_aug_frame(frame_root, frame_indices[i]) for i in range(sample_num)], dim=0)
        return frames, cls


@DATASETS.register_module()
class KineticsClipFolderDatasetV2Pair(KineticsClipFolderDatasetV2):
    def __init__(self, root, transform=None, split='train_list'):
        super(KineticsClipFolderDatasetV2Pair, self).__init__(root, split)
        self.transform = transform
        assert self.transform is not None

    def __getitem__(self, item):
        frame_root, frame_num, cls = self.samples[item]
        rand_segment = random.randint(0, 1)
        if rand_segment == 0:
            frame_idx_1 = random.randint(1, frame_num // 2)
            frame_idx_2 = random.randint(frame_num // 2 + 1, frame_num)
        else:
            frame_idx_2 = random.randint(1, frame_num // 2)
            frame_idx_1 = random.randint(frame_num // 2 + 1, frame_num)
        frame_aug1 = self._get_aug_frame(frame_root, frame_idx_1)
        frame_aug2 = self._get_aug_frame(frame_root, frame_idx_2)
        return frame_aug1, frame_aug2


@DATASETS.register_module()
class KineticsClipFolderDatasetV2Triplet(KineticsClipFolderDatasetV2):
    def __init__(self, root, transform=None, split='train_list'):
        super(KineticsClipFolderDatasetV2Triplet, self).__init__(root, split)
        self.transform = transform
        assert self.transform is not None

    def __getitem__(self, item):
        frame_root, frame_num, cls = self.samples[item]
        rand_segment = random.randint(0, 1)
        if rand_segment == 0:
            frame_idx_1 = random.randint(1, frame_num // 2)
            frame_idx_2 = random.randint(frame_num // 2 + 1, frame_num)
        else:
            frame_idx_2 = random.randint(1, frame_num // 2)
            frame_idx_1 = random.randint(frame_num // 2 + 1, frame_num)
        frame1_aug1 = self._get_aug_frame(frame_root, frame_idx_1)
        frame1_aug2 = self._get_aug_frame(frame_root, frame_idx_1)
        frame2_aug = self._get_aug_frame(frame_root, frame_idx_2)
        return frame1_aug1, frame1_aug2, frame2_aug


@DATASETS.register_module()
class KineticsClipFolderDatasetV2Order(KineticsClipFolderDatasetV2):
    def __init__(self, root, transform=None, split='train_list'):
        super(KineticsClipFolderDatasetV2Order, self).__init__(root, split)
        self.transform = transform
        assert self.transform is not None

    def __getitem__(self, item):
        frame_root, frame_num, cls = self.samples[item]
        rand_segment = random.randint(0, 1)
        if rand_segment == 0:
            frame_idx_1 = random.randint(1, frame_num // 2)
            frame_idx_2 = random.randint(frame_num // 2 + 1, frame_num)
            frame_idx_3 = random.randint(frame_num // 2 + 1, frame_num)
        else:
            frame_idx_1 = random.randint(frame_num // 2 + 1, frame_num)
            frame_idx_2 = random.randint(1, frame_num // 2)
            frame_idx_3 = random.randint(1, frame_num // 2)
        if frame_idx_2 > frame_idx_3:
            frame_idx_2, frame_idx_3 = frame_idx_3, frame_idx_2
        frame1_aug1 = self._get_aug_frame(frame_root, frame_idx_1)
        frame1_aug2 = self._get_aug_frame(frame_root, frame_idx_1)
        frame2_aug = self._get_aug_frame(frame_root, frame_idx_2)
        frame3_aug = self._get_aug_frame(frame_root, frame_idx_3)
        return frame1_aug1, frame1_aug2, frame2_aug, frame3_aug, rand_segment

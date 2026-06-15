import os
import numpy as np
import scipy.io as sio
from torch.utils.data import Dataset
import torch
import cv2

def detect_coherency_variable(mat_file_path):

    data = sio.loadmat(mat_file_path)
    variables = [k for k in data.keys() if not k.startswith('__')]
    for var in variables:
        var_lower = var.lower()
        if var_lower in ['feat_9ch', 'coherency', 't', 'data', 'feature', 'coherence']:
            return var
    for var in variables:
        if isinstance(data[var], np.ndarray) and data[var].size > 0:
            return var
    return variables[0]

def detect_variable_name(mat_file_path):
    """检测 .mat 文件中的主要变量名"""
    data = sio.loadmat(mat_file_path)
    variables = [k for k in data.keys() if not k.startswith('__')]

    for var in variables:
        if any(k in var for k in ['Ps', 'Pv', 'Pd', 'power', 'T11']):
            return var
    for var in variables:
        if isinstance(data[var], np.ndarray) and data[var].size > 0:
            return var
    return variables[0]


class PolSARDataset(Dataset):
    def __init__(self, coherency_path, freeman_paths, label_path, patch_size=15, stride=8,
                 transform=True, is_train=True, samples_per_class=500):
        coherency_var = detect_coherency_variable(coherency_path)
        coherency_data = sio.loadmat(coherency_path)[coherency_var].astype(np.float32)
        if coherency_data.ndim == 3:
            if coherency_data.shape[0] == 9:
                pass
            elif coherency_data.shape[-1] == 9:
                coherency_data = np.transpose(coherency_data, (2, 0, 1))

        self.main_features = np.log1p(np.abs(coherency_data))
        for c in range(9):
            mean, std = self.main_features[c].mean(), self.main_features[c].std() + 1e-8
            self.main_features[c] = (self.main_features[c] - mean) / std

        freeman_channels = []
        for path in freeman_paths:
            var_name = detect_variable_name(path)
            data = sio.loadmat(path)[var_name].astype(np.float32)
            freeman_channels.append(data)
        self.phy_features = np.stack(freeman_channels, axis=-1)  # (H, W, 3)
        self.phy_gt = np.argmax(self.phy_features, axis=-1).astype(np.int64)

        # 标签处理
        raw_label = sio.loadmat(label_path)['label'].astype(np.int64)
        self.original_height, self.original_width = raw_label.shape
        unique_labels = np.unique(raw_label)
        valid_classes = [cls for cls in unique_labels if cls not in [0, 255]]
        self.class_mapping = {orig: new for new, orig in enumerate(sorted(valid_classes))}
        self.num_classes = len(valid_classes)
        self.ignore_index = 255

        self.label = np.full_like(raw_label, self.ignore_index)
        for orig, new in self.class_mapping.items():
            self.label[raw_label == orig] = new

        self.patch_size = patch_size
        self.is_train = is_train
        self.transform = transform

        if is_train:
            self.coords = self._generate_balanced_coords(samples_per_class)
        else:
            self.stride = stride
            self.coords = self._generate_sliding_coords()

    def _generate_balanced_coords(self, samples_per_class):
        coords = []
        ps = self.patch_size
        half_ps = ps // 2
        H, W = self.label.shape

        for cls_idx in range(self.num_classes):
            y_indices, x_indices = np.where(self.label == cls_idx)
            num_available = len(y_indices)

            if num_available == 0:
                continue
            replace = num_available < samples_per_class
            choice_indices = np.random.choice(num_available, samples_per_class, replace=replace)

            for idx in choice_indices:
                cy, cx = y_indices[idx], x_indices[idx]
                y1 = max(0, min(cy - half_ps, H - ps))
                x1 = max(0, min(cx - half_ps, W - ps))
                coords.append((y1, x1))

        return coords

    def _generate_sliding_coords(self):
        ps, H, W = self.patch_size, self.original_height, self.original_width
        coords = []
        for i in range(0, H - ps + 1, self.stride):
            for j in range(0, W - ps + 1, self.stride):
                coords.append((i, j))
        return coords

    def __len__(self):
        return len(self.coords)

    def __getitem__(self, idx):
        i, j = self.coords[idx]
        ps = self.patch_size

        main_patch = self.main_features[:, i:i+ps, j:j+ps]
        phy_patch = self.phy_features[i:i+ps, j:j+ps, :]
        phy_patch = np.transpose(phy_patch, (2, 0, 1))
        sem_patch = self.label[i:i+ps, j:j+ps]
        phy_gt_patch = self.phy_gt[i:i+ps, j:j+ps]

        if self.is_train and self.transform:
            if np.random.random() > 0.5:  # 水平翻转
                main_patch = np.flip(main_patch, axis=2).copy()
                phy_patch = np.flip(phy_patch, axis=1).copy()
                sem_patch = np.flip(sem_patch, axis=1).copy()
                phy_gt_patch = np.flip(phy_gt_patch, axis=1).copy()
            if np.random.random() > 0.5:  # 垂直翻转
                main_patch = np.flip(main_patch, axis=1).copy()
                phy_patch = np.flip(phy_patch, axis=0).copy()
                sem_patch = np.flip(sem_patch, axis=0).copy()
                phy_gt_patch = np.flip(phy_gt_patch, axis=0).copy()

        sample = {
            'main': torch.from_numpy(main_patch).float(),
            'phy': torch.from_numpy(phy_patch).float(),
            'label': torch.from_numpy(sem_patch).long(),
            'phy_gt': torch.from_numpy(phy_gt_patch).long()
        }
        return sample
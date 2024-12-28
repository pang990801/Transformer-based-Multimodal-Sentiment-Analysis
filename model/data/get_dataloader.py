"""Implements dataloaders for AFFECT data."""
import os
import sys
from typing import *
import pickle
import h5py
import numpy as np
from torch.nn import functional as F

sys.path.append(os.getcwd())

import torch
import torchtext as text
from collections import defaultdict
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset


np.seterr(divide='ignore', invalid='ignore')

def drop_entry(dataset):
    """Drop entries where there's no text in the data."""
    drop = []
    for ind, k in enumerate(dataset["text"]):
        if k.sum() == 0:
            drop.append(ind)
    for modality in list(dataset.keys()):
        dataset[modality] = np.delete(dataset[modality], drop, 0)
    return dataset


def z_norm(dataset, max_seq_len=50):
    """Normalize data in the dataset."""
    processed = {}
    text = dataset['text'][:, :max_seq_len, :]
    vision = dataset['vision'][:, :max_seq_len, :]
    audio = dataset['audio'][:, :max_seq_len, :]
    for ind in range(dataset["text"].shape[0]):
        vision[ind] = np.nan_to_num(
            (vision[ind] - vision[ind].mean(0, keepdims=True)) / (np.std(vision[ind], axis=0, keepdims=True)))
        audio[ind] = np.nan_to_num(
            (audio[ind] - audio[ind].mean(0, keepdims=True)) / (np.std(audio[ind], axis=0, keepdims=True)))
        text[ind] = np.nan_to_num(
            (text[ind] - text[ind].mean(0, keepdims=True)) / (np.std(text[ind], axis=0, keepdims=True)))

    processed['vision'] = vision
    processed['audio'] = audio
    processed['text'] = text
    processed['labels'] = dataset['labels']
    return processed


class Affectdataset(Dataset):
    def __init__(self, data: Dict, flatten_time_series: bool, aligned: bool = True, task: str = None, max_pad=False,
                 max_pad_num=50, data_type='mosei', z_norm=False) -> None:

        self.dataset = data
        self.flatten = flatten_time_series
        self.aligned = aligned
        self.task = task
        self.max_pad = max_pad
        self.max_pad_num = max_pad_num
        self.data_type = data_type
        self.z_norm = z_norm
        self.dataset['audio'][self.dataset['audio'] == -np.inf] = 0.0

    def __getitem__(self, ind):

        vision = torch.tensor(self.dataset['vision'][ind])
        audio = torch.tensor(self.dataset['audio'][ind])
        text = torch.tensor(self.dataset['text'][ind])

        if self.aligned:
            try:
                start = text.nonzero(as_tuple=False)[0][0]
                # start = 0
            except:
                print(text, ind)
                exit()
            vision = vision[start:].float()
            audio = audio[start:].float()
            text = text[start:].float()
        else:
            vision = vision[vision.nonzero()[0][0]:].float()
            audio = audio[audio.nonzero()[0][0]:].float()
            text = text[text.nonzero()[0][0]:].float()

        # z-normalize data
        if self.z_norm:
            vision = torch.nan_to_num(
                (vision - vision.mean(0, keepdims=True)) / (torch.std(vision, axis=0, keepdims=True)))
            audio = torch.nan_to_num((audio - audio.mean(0, keepdims=True)) / (torch.std(audio, axis=0, keepdims=True)))
            text = torch.nan_to_num((text - text.mean(0, keepdims=True)) / (torch.std(text, axis=0, keepdims=True)))

        def _get_class(flag, data_type=self.data_type):
            if data_type in ['mosi', 'mosei', 'sarcasm']:
                if flag > 0:
                    return [[1]]
                else:
                    return [[0]]
            else:
                return [flag]

        tmp_label = self.dataset['labels'][ind]

        label = torch.tensor(tmp_label).float()

        if self.flatten:
            return [vision.flatten(), audio.flatten(), text.flatten(), ind, \
                    label]
        else:
            if self.max_pad:
                tmp = [vision, audio, text, label]
                for i in range(len(tmp) - 1):
                    tmp[i] = tmp[i][:self.max_pad_num]
                    tmp[i] = F.pad(tmp[i], (0, 0, 0, self.max_pad_num - tmp[i].shape[0]))
            else:
                tmp = [vision, audio, text, ind, label]
            return tmp
    def __len__(self):
        """Get length of dataset."""
        return self.dataset['vision'].shape[0]

def get_mosi_dataloader(
        filepath: str, batch_size: int = 32, max_seq_len=50, max_pad=False,
        num_workers: int = 2, flatten_time_series: bool = False, task=None, data_type='mosi', z_norm=False) -> DataLoader:

    with open(filepath, "rb") as f:
        alldata = pickle.load(f)

    processed_dataset = {'train': {}, 'test': {}, 'valid': {}}
    alldata['train'] = drop_entry(alldata['train'])
    alldata['valid'] = drop_entry(alldata['valid'])
    alldata['test'] = drop_entry(alldata['test'])


    merged_test_data = {}
    for key in alldata['test'].keys():
        # 连接train、valid和test的数据
        merged_test_data[key] = np.concatenate([
            alldata['train'][key],
            alldata['valid'][key],
            alldata['test'][key]
        ], axis=0)

    process = eval("_process_2") if max_pad else eval("_process_1")


    for dataset in alldata:
        processed_dataset[dataset] = alldata[dataset]

    test = DataLoader(Affectdataset(merged_test_data, flatten_time_series, task=task, max_pad=max_pad,
                                    max_pad_num=max_seq_len, data_type=data_type, z_norm=z_norm), shuffle=False, num_workers=num_workers, batch_size=batch_size,
                      collate_fn=process)

    return test

def get_ablation_mosi_dataloader(
        filepath: str, batch_size: int = 32, max_seq_len=50, max_pad=False, modalities=None,
        num_workers: int = 2, flatten_time_series: bool = False, task=None, data_type='mosi',embedding='bert', z_norm=False) -> DataLoader:

    with open(filepath, "rb") as f:
        alldata = pickle.load(f)

    processed_dataset = {'train': {}, 'test': {}, 'valid': {}}
    alldata['train'] = drop_entry(alldata['train'])
    alldata['valid'] = drop_entry(alldata['valid'])
    alldata['test'] = drop_entry(alldata['test'])

    merged_test_data = {}
    for key in alldata['test'].keys():  # 假设所有数据集都有相同的键
        # 连接train、valid和test的数据
        merged_test_data[key] = np.concatenate([
            alldata['train'][key],
            alldata['valid'][key],
            alldata['test'][key]
        ], axis=0)

    process = eval("_process_2") if max_pad else eval("_process_1")



    for dataset in alldata:
        processed_dataset[dataset] = alldata[dataset]

    def filter_modalities_list(dataset, modalities):

        # 定义模态索引映射及其对应的维度
        modality_index_map = {'visual': 0, 'audio': 1, 'text': 2}
        if embedding == 'bert':
            modality_dims = {'visual': (50, 35), 'audio': (50, 74), 'text': (50, 768)}
        if embedding == 'glove':
            modality_dims = {'visual': (50, 35), 'audio': (50, 74), 'text': (50, 300)}# 假设每列数据固定维度
        selected_indices = [modality_index_map[modality] for modality in modalities]

        # 初始化新的数据集
        filtered_data = []
        for row in dataset:
            # 构建新的行数据
            filtered_row = []
            for idx, modality_name in enumerate(modality_index_map.keys()):
                if idx in selected_indices:
                    filtered_row.append(row[idx])  # 保留需要的模态
                else:
                    filtered_row.append(torch.zeros(modality_dims[modality_name]))  # 未选中的模态设置为零张量

            # 保留 id 列
            filtered_row.append(row[3])
            filtered_data.append(filtered_row)

        return filtered_data


    test_dataset = Affectdataset(merged_test_data, flatten_time_series, task=task, max_pad=max_pad,
                                    max_pad_num=max_seq_len, data_type=data_type, z_norm=z_norm)
    test_dataset = filter_modalities_list(test_dataset, modalities=modalities)


    test = DataLoader(test_dataset, \
                      shuffle=False, num_workers=num_workers, batch_size=batch_size, \
                      collate_fn=process)
    return test

def get_dataloader(
        filepath: str, batch_size: int = 32, max_seq_len=50, max_pad=False, train_shuffle: bool = True,
        num_workers: int = 2, flatten_time_series: bool = False, task=None, robust_test=False, data_type='mosi',
        raw_path='/home/van/backup/pack/mosi/mosi.hdf5', z_norm=False) -> DataLoader:

    with open(filepath, "rb") as f:
        alldata = pickle.load(f)

    processed_dataset = {'train': {}, 'test': {}, 'valid': {}}
    alldata['train'] = drop_entry(alldata['train'])
    alldata['valid'] = drop_entry(alldata['valid'])
    alldata['test'] = drop_entry(alldata['test'])

    process = eval("_process_2") if max_pad else eval("_process_1")


    for dataset in alldata:
        processed_dataset[dataset] = alldata[dataset]

    train = DataLoader(Affectdataset(processed_dataset['train'], flatten_time_series, task=task, max_pad=max_pad, max_pad_num=max_seq_len, data_type=data_type, z_norm=z_norm), \
                       shuffle=train_shuffle, num_workers=num_workers, batch_size=batch_size, \
                       collate_fn=process)
    valid = DataLoader(Affectdataset(processed_dataset['valid'], flatten_time_series, task=task, max_pad=max_pad, max_pad_num=max_seq_len, data_type=data_type, z_norm=z_norm), \
                       shuffle=False, num_workers=num_workers, batch_size=batch_size, \
                       collate_fn=process)
    test = DataLoader(Affectdataset(processed_dataset['test'], flatten_time_series, task=task, max_pad=max_pad, max_pad_num=max_seq_len, data_type=data_type, z_norm=z_norm), \
                      shuffle=False, num_workers=num_workers, batch_size=batch_size, \
                      collate_fn=process)
    return train, valid, test


def get_ablation_dataloader(
        filepath: str, batch_size: int = 32, max_seq_len=50, max_pad=False, train_shuffle: bool = True, modalities=None,
        num_workers: int = 2, flatten_time_series: bool = False, task=None, data_type='mosi',embedding='bert', z_norm=False) -> DataLoader:

    with open(filepath, "rb") as f:
        alldata = pickle.load(f)

    processed_dataset = {'train': {}, 'test': {}, 'valid': {}}
    alldata['train'] = drop_entry(alldata['train'])
    alldata['valid'] = drop_entry(alldata['valid'])
    alldata['test'] = drop_entry(alldata['test'])

    process = eval("_process_2") if max_pad else eval("_process_1")


    for dataset in alldata:
        processed_dataset[dataset] = alldata[dataset]

    def filter_modalities_list(dataset, modalities):
        """
        根据提供的模态列表筛选数据集中的模态（适用于列表结构的数据），未选中的模态设置为全零张量。

        Args:
            dataset (Dataset): 输入数据集，如 train_dataset, valid_dataset, test_dataset。
            modalities (list): 需要保留的模态列表，例如 ['text', 'audio']。

        Returns:
            Dataset: 只包含指定模态的数据集，未选中的模态被设置为全零张量。
        """
        # 定义模态索引映射及其对应的维度
        modality_index_map = {'visual': 0, 'audio': 1, 'text': 2}
        if embedding == 'bert':
            modality_dims = {'visual': (50, 35), 'audio': (50, 74), 'text': (50, 768)}
        if embedding == 'glove':
            modality_dims = {'visual': (50, 35), 'audio': (50, 74), 'text': (50, 300)}# 假设每列数据固定维度
        selected_indices = [modality_index_map[modality] for modality in modalities]

        # 初始化新的数据集
        filtered_data = []
        for row in dataset:
            # 构建新的行数据
            filtered_row = []
            for idx, modality_name in enumerate(modality_index_map.keys()):
                if idx in selected_indices:
                    filtered_row.append(row[idx])  # 保留需要的模态
                else:
                    filtered_row.append(torch.zeros(modality_dims[modality_name]))  # 未选中的模态设置为零张量

            # 保留 id 列
            filtered_row.append(row[3])  # 假设 id 始终是第 4 列
            filtered_data.append(filtered_row)

        return filtered_data

    train_dataset = Affectdataset(processed_dataset['train'], flatten_time_series, task=task, max_pad=max_pad,
                        max_pad_num=max_seq_len, data_type=data_type, z_norm=z_norm)
    train_dataset = filter_modalities_list(train_dataset, modalities=modalities)

    valid_dataset = Affectdataset(processed_dataset['valid'], flatten_time_series, task=task, max_pad=max_pad,
                                     max_pad_num=max_seq_len, data_type=data_type, z_norm=z_norm)
    valid_dataset = filter_modalities_list(valid_dataset, modalities=modalities)

    test_dataset = Affectdataset(processed_dataset['test'], flatten_time_series, task=task, max_pad=max_pad,
                                    max_pad_num=max_seq_len, data_type=data_type, z_norm=z_norm)
    test_dataset = filter_modalities_list(test_dataset, modalities=modalities)


    train = DataLoader(train_dataset, \
                       shuffle=train_shuffle, num_workers=num_workers, batch_size=batch_size, \
                       collate_fn=process)
    valid = DataLoader(valid_dataset, \
                       shuffle=False, num_workers=num_workers, batch_size=batch_size, \
                       collate_fn=process)
    test = DataLoader(test_dataset, \
                      shuffle=False, num_workers=num_workers, batch_size=batch_size, \
                      collate_fn=process)
    return train, valid, test




def _process_1(inputs: List):
    processed_input = []
    processed_input_lengths = []
    inds = []
    labels = []

    for i in range(len(inputs[0]) - 2):
        feature = []
        for sample in inputs:
            feature.append(sample[i])
        processed_input_lengths.append(torch.as_tensor([v.size(0) for v in feature]))
        pad_seq = pad_sequence(feature, batch_first=True)
        processed_input.append(pad_seq)

    for sample in inputs:
        inds.append(sample[-2])
        # if len(sample[-2].shape) > 2:
        #     labels.append(torch.where(sample[-2][:, 1] == 1)[0])
        # else:
        if sample[-1].shape[1] > 1:
            labels.append(sample[-1].reshape(sample[-1].shape[1], sample[-1].shape[0])[0])
        else:
            labels.append(sample[-1])

    return processed_input, processed_input_lengths, \
           torch.tensor(inds).view(len(inputs), 1), torch.tensor(labels).view(len(inputs), 1)


def _process_2(inputs: List):
    processed_input = []
    processed_input_lengths = []
    labels = []

    for i in range(len(inputs[0]) - 1):
        feature = []
        for sample in inputs:
            feature.append(sample[i])
        processed_input_lengths.append(torch.as_tensor([v.size(0) for v in feature]))
        # pad_seq = pad_sequence(feature, batch_first=True)
        processed_input.append(torch.stack(feature))

    for sample in inputs:
        
        # if len(sample[-2].shape) > 2:
        #     labels.append(torch.where(sample[-2][:, 1] == 1)[0])
        # else:
        # print(sample[-1].shape)
        if sample[-1].shape[1] > 1:
            labels.append(sample[-1].reshape(sample[-1].shape[1], sample[-1].shape[0])[0])
        else:
            labels.append(sample[-1])

    return processed_input[0], processed_input[1], processed_input[2], torch.tensor(labels).view(len(inputs), 1)


            

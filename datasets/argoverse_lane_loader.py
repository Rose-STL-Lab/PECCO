"Functions loading the .pkl version preprocessed data"
from glob import glob
import pickle
import os
import math
import numpy as np
from typing import Any, Dict, List, Tuple, Union
#from argoverse.map_representation.map_api import ArgoverseMap
import torch
from torch.utils.data import IterableDataset, DataLoader


class ArgoverseDataset(IterableDataset):
    def __init__(self, data_path: str, 
                 max_lane_nodes=650, min_lane_nodes=0, rotate=False, cannon=False, shuffle=True):
        super(ArgoverseDataset, self).__init__()
        self.data_path = data_path
        self.rotate = rotate
        self.cannon = cannon
        self.pkl_list = glob(os.path.join(self.data_path, '*'))
        if shuffle:
            np.random.shuffle(self.pkl_list)
        else:
            self.pkl_list.sort()
        self.max_lane_nodes = max_lane_nodes
        self.min_lane_nodes = min_lane_nodes
        
    def __len__(self):
        return len(self.pkl_list)
    
    def __iter__(self):
        # pkl_path = self.pkl_list[idx]
        for pkl_path in self.pkl_list:
            with open(pkl_path, 'rb') as f:
                data = pickle.load(f)
            # data = {k:v[0] for k, v in data.items()}
            lane_mask = np.zeros(self.max_lane_nodes, dtype=np.float32)
            lane_mask[:len(data['lane'])] = 1.0
            data['lane_mask'] = [lane_mask]
            data['lane'] = np.array(data['lane'][0])
            data['lane_norm'] = np.array(data['lane_norm'][0])

            if data['lane'].shape[0] > self.max_lane_nodes:
                continue

            if data['lane'].shape[0] < self.min_lane_nodes:
                continue

            data['lane'] = [self.expand_particle(data['lane'][...,:2], self.max_lane_nodes, 0)]
            data['lane_norm'] = [self.expand_particle(data['lane_norm'][...,:2], self.max_lane_nodes, 0)]

            if self.rotate:
                theta = (np.random.rand(1) * 2 * np.pi)[0]
                convert_keys = (['pos' + str(i) for i in range(30)] + 
                                ['vel' + str(i) for i in range(30)] + 
                                ['pos_2s', 'vel_2s', 'lane', 'lane_norm'])
                
                for k in convert_keys:
                    data[k] = [rotation(theta, data[k][0])]
            
            if self.cannon:
                trackid = np.expand_dims(data['track_id0'][0], -1)
                agent = (trackid == data['agent_id']).nonzero()[0][0]
                arctan = data['vel0'][0][agent,1]/data['vel0'][0][agent,0]
                if math.isnan(arctan):
                    theta = 0
                    print(data['vel0'][0][agent,:])
                else:
                    theta = np.arctan(arctan)
                convert_keys = (['pos' + str(i) for i in range(30)] + 
                                ['vel' + str(i) for i in range(30)] + 
                                ['pos_2s', 'vel_2s', 'lane', 'lane_norm'])
                
                for k in convert_keys:
                    data[k] = [rotation(theta, data[k][0])]

            yield data
    
    @classmethod
    def expand_particle(cls, arr, max_num, axis, value_type='int'):
        dummy_shape = list(arr.shape)
        dummy_shape[axis] = max_num - arr.shape[axis]
        dummy = np.zeros(dummy_shape)
        if value_type == 'str':
            dummy = np.array(['dummy' + str(i) for i in range(np.product(dummy_shape))]).reshape(dummy_shape)
        return np.concatenate([arr, dummy], axis=axis)
    
    
def cat_key(data, key):
    result = []
    for d in data:
        toappend = d[key]
        if not isinstance(toappend,list):
            result += [toappend]
        else:
            result += toappend
    return result


def dict_collate_func(data):
    keys = data[0].keys()
    data = {key: cat_key(data, key) for key in keys}
    return data


def read_pkl_data(data_path: str, batch_size: int, 
                  shuffle: bool=False, repeat: bool=False, **kwargs):
    dataset = ArgoverseDataset(data_path=data_path, shuffle=shuffle, **kwargs)
    loader = DataLoader(dataset, batch_size=int(batch_size), collate_fn=dict_collate_func)

    if repeat:
        while True:
            for data in loader:
                yield data
    else:
        for data in loader:
            yield data

def RotMat(theta):
    m = np.array([
            [np.cos(theta), -np.sin(theta)], 
            [np.sin(theta), np.cos(theta)]
        ])
    return m

def rotation(theta, field):
    rotmat = RotMat(theta)
    rot_field = np.zeros(field.shape)
    rot_field[...,:2] =  np.einsum('ij,...j->...i', rotmat, field[...,:2])
    if field.shape[-1] > 2:
        rot_field[...,2] = 0
    return rot_field
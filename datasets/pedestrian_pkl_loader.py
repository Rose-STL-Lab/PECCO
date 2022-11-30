"Functions loading the .pkl version preprocessed data"
from tensorpack import dataflow
from glob import glob
import numpy as np
import pickle
import os

class PedestrainPklLoader(dataflow.RNGDataFlow):
    def __init__(self, data_path: str, shuffle: bool=True, max_num=40, rotate=False):
        super(PedestrainPklLoader, self).__init__()
        self.data_path = data_path
        self.shuffle = shuffle
        self.max_num = max_num
        self.rotate = rotate
        
    def __iter__(self):
        pkl_list = glob(os.path.join(self.data_path, '*'))
        pkl_list.sort()
        if self.shuffle:
            self.rng.shuffle(pkl_list)
            
        for pkl_path in pkl_list:
            with open(pkl_path, 'rb') as f:
                data = pickle.load(f)
            data['man_mask'] = data['man_mask'][:self.max_num]
            if sum(data['man_mask']) > self.max_num:
                continue
            if 'pos12' not in data.keys():
                continue

            if self.rotate:
                theta = (np.random.rand(1) * 2 * np.pi)[0]
                convert_keys = (['pos' + str(i) for i in range(13)] + 
                                ['vel' + str(i) for i in range(13)] + 
                                ['pos_enc', 'vel_enc'])
                
                for k in convert_keys:
                    data[k] = rotation(theta, data[k])
            yield data



def read_pkl_data(data_path: str, batch_size: int, 
                  shuffle: bool=False, repeat: bool=False, **kwargs):
    df = PedestrainPklLoader(data_path=data_path, shuffle=shuffle, **kwargs)
    if repeat:
        df = dataflow.RepeatedData(df, -1)
    df = dataflow.BatchData(df, batch_size=batch_size, use_list=True)
    df.reset_state()
    return df

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

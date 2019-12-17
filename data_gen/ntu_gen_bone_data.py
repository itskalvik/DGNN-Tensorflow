import os
import numpy as np
from numpy.lib.format import open_memmap
from tqdm import tqdm

directed_edges = [(i-1, j-1) for i, j in [
    (1, 13), (1, 17), (2, 1), (3, 4), (5, 6),
    (6, 7), (7, 8), (8, 22), (8, 23), (9, 10),
    (10, 11), (11, 12), (12, 24), (12, 25), (13, 14),
    (14, 15), (15, 16), (17, 18), (18, 19), (19, 20),
    (21, 2), (21, 3), (21, 5), (21, 9)
]]

sets = {'train', 'val'}
datasets = {'ntu/xview', 'ntu/xsub'}

def gen_bone_data():
    """Generate bone data from joint data for NTU skeleton dataset"""
    for dataset in datasets:
        for set in sets:
            print(dataset, set)
            data = np.load('../data/{}/{}_data_joint.npy'.format(dataset, set))
            N, C, T, V, M = data.shape
            fp_sp = open_memmap(
                '../data/{}/{}_data_bone.npy'.format(dataset, set),
                dtype='float32',
                mode='w+',
                shape=(N, 3, T, len(directed_edges), M))

            for edge_id, (source_node, target_node) in tqdm(enumerate(directed_edges)):
                # Assign bones to be joint1 - joint2, the pairs are pre-determined and hardcoded
                # There also happens to be 25 bones
                fp_sp[:, :, :, edge_id, :] = data[:, :, :, source_node, :] - data[:, :, :, target_node, :]


if __name__ == '__main__':
    gen_bone_data()

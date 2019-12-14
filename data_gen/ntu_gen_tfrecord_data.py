import os
import pickle
import argparse
import numpy as np
from tqdm import tqdm
import tensorflow as tf

sets = {'train', 'val'}
datasets = {'ntu/xview', 'ntu/xsub'}
streams = {'', '_motion'}


def _bytes_feature(value):
  """Returns a bytes_list from a string / byte."""
  if isinstance(value, type(tf.constant(0))):
    value = value.numpy() # BytesList won't unpack a string from an EagerTensor.
  return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def _int64_feature(value):
  """Returns an int64_list from a bool / enum / int / uint."""
  return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

def serialize_example(bone_data, joint_data, label):
    feature = {
        'bone_data' : _bytes_feature(tf.io.serialize_tensor(bone_data.astype(np.float32))),
        'joint_data': _bytes_feature(tf.io.serialize_tensor(joint_data.astype(np.float32))),
        'label'     : _int64_feature(label)
    }

    return tf.train.Example(features=tf.train.Features(feature=feature)).SerializeToString()

def gen_tfrecord_data(num_shards_train, num_shards_val):
    for dataset in datasets:
        for set in sets:
            for stream in streams:
                print("Generating TFRecord file for {} {} {}".format(dataset, set, stream))

                label_path = '../data/{}/{}_label.pkl'.format(dataset, set)
                bone_data_path = '../data/{}/{}_data_bone{}.npy'.format(dataset, set, stream)
                joint_data_path = '../data/{}/{}_data_joint{}.npy'.format(dataset, set, stream)

                if not (os.path.exists(label_path) and \
                        os.path.exists(bone_data_path) and \
                        os.path.exists(joint_data_path)):
                    print('Joint/Bone/Label data does not exist for {} {} set'.format(dataset, set))
                    return

                try:
                    with open(label_path) as f:
                        _, labels = pickle.load(f)
                except:
                    # for pickle file from python2
                    with open(label_path, 'rb') as f:
                        _, labels = pickle.load(f, encoding='latin1')

                # Datashape: Total_samples, 3, 300, 25, 2
                bone_data  = np.load(bone_data_path, mmap_mode='r')
                joint_data = np.load(joint_data_path, mmap_mode='r')

                # Datashape: Total_samples, 2, 300, 25, 3
                bone_data  = np.swapaxes(bone_data, 1, -1)
                joint_data = np.swapaxes(joint_data, 1, -1)

                if not (os.path.exists('../data/{0}/{1}{2}_data'.format(dataset, set, stream))):
                    os.mkdir('../data/{0}/{1}{2}_data'.format(dataset, set, stream))

                # Loop through samples and insert into tfrecord
                if "val" in set:
                    step = len(labels)//num_shards_val
                    num_shards = num_shards_val
                elif "train" in set:
                    step = len(labels)//num_shards_train
                    num_shards = num_shards_train

                for shard in tqdm(range(num_shards)):
                    tfrecord_data_path = '../data/{0}/{1}{2}_data/{1}{2}_data_{3}.tfrecord'.format(dataset, set, stream, shard)
                    with tf.io.TFRecordWriter(tfrecord_data_path) as writer:
                        for i in range(shard*step, (shard*step)+step if shard < num_shards_train-1 else len(labels)):
                            writer.write(serialize_example(bone_data[i], joint_data[i], labels[i]))

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='NTU-RGB-D Data TFRecord Converter.')
    parser.add_argument('--num-shards-train', type=int, default=40, help='number of files to split train dataset into')
    parser.add_argument('--num-shards-val', type=int, default=1, help='number of files to split val dataset into')
    arg = parser.parse_args()
    gen_tfrecord_data(arg.num_shards_train, arg.num_shards_val)

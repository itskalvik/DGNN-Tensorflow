import os
import pickle
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

def gen_tfrecord_data():
    for dataset in datasets:
        for set in sets:
            for stream in streams:
                print("Generating TFRecord file for {} {} {}".format(dataset, set, stream))

                label_path = '../data/{}/{}_label.pkl'.format(dataset, set)
                bone_data_path = '../data/{}/{}_data_bone{}.npy'.format(dataset, set, stream)
                joint_data_path = '../data/{}/{}_data_joint{}.npy'.format(dataset, set, stream)
                tfrecord_data_path = '../data/{}/{}{}_data.tfrecord'.format(dataset, set, stream)

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

                # Loop through samples and insert into tfrecord
                with tf.io.TFRecordWriter(tfrecord_data_path) as writer:
                    for i in tqdm(range(len(labels))):
                        writer.write(serialize_example(bone_data[i], joint_data[i], labels[i]))

if __name__ == '__main__':
    gen_tfrecord_data()

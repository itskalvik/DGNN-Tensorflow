from model.dgnn import DGNN
import tensorflow as tf
from tqdm import tqdm
import argparse
import inspect
import shutil
import yaml
import os

from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession

config = ConfigProto()
config.gpu_options.allow_growth = True
session = InteractiveSession(config=config)


def get_parser():
    # parameter priority: command line > config > default
    parser = argparse.ArgumentParser(
        description='Directed Graph Neural Net for Skeleton Action Recognition')
    parser.add_argument(
        '--base-lr', type=float, default=1e-2, help='initial learning rate')
    parser.add_argument(
        '--num-classes', type=int, default=60, help='number of classes in dataset')
    parser.add_argument(
        '--batch-size', type=int, default=16, help='training batch size')
    parser.add_argument(
        '--num-epochs', type=int, default=120, help='total epochs to train')
    parser.add_argument(
        '--save-freq', type=int, default=25, help='periodicity of saving model weights')
    parser.add_argument(
        '--checkpoint-path',
        default="checkpoints/DGNN",
        help='folder to store model weights')
    parser.add_argument(
        '--log-dir',
        default="logs/DGNN",
        help='folder to store model-definition/training-logs/hyperparameters')
    parser.add_argument(
        '--train-data-path',
        default="data/ntu/xview/train_data.tfrecord",
        help='path to tfrecord file with training dataset')
    parser.add_argument(
        '--test-data-path',
        default="data/ntu/xview/val_data.tfrecord",
        help='path to tfrecord file with testing dataset')
    parser.add_argument(
        '--freeze-graph-until',
        type=int,
        default=10,
        help='number of epochs before making graphs learnable')
    return parser


def save_arg(arg):
    # save arg
    arg_dict = vars(arg)
    if not os.path.exists(arg.log_dir):
        os.makedirs(arg.log_dir)
    with open(os.path.join(arg.log_dir, "config.yaml"), 'w') as f:
        yaml.dump(arg_dict, f)


'''
get_dataset: Returns a tensorflow dataset object with joint, bone and one hot encoded label data
Args:
  filepath        : File path to TFRecord file with dataset
  num_classes     : Number of classes in dataset for one hot encoding
  batch_size      : Represents the number of consecutive elements of this dataset to combine in a single batch.
  drop_remainder  : If True, the last batch will be dropped in the case it has fewer than batch_size elements. Defaults to False
  shuffle         : If True, the data samples will be shuffled randomly. Defaults to False
Returns:
  The Dataset with joint, bone and one hot encoded label data
'''
def get_dataset(filepath, num_classes=60, batch_size=32, drop_remainder=False, shuffle=False):
    # dictionary describing the features.
    feature_description = {
        'bone_data' : tf.io.FixedLenFeature([], tf.string),
        'joint_data': tf.io.FixedLenFeature([], tf.string),
        'label'     : tf.io.FixedLenFeature([], tf.int64)
    }

    # parse each proto and, the bone and joint features within
    def _parse_feature_function(example_proto):
        features = tf.io.parse_single_example(example_proto, feature_description)
        return tf.io.parse_tensor(features['bone_data'], tf.float32),  \
               tf.io.parse_tensor(features['joint_data'], tf.float32), \
               tf.one_hot(features['label'], num_classes)

    dataset = tf.data.TFRecordDataset(filepath)
    dataset = dataset.map(_parse_feature_function)
    dataset = dataset.batch(batch_size, drop_remainder=drop_remainder)
    dataset = dataset.prefetch(batch_size)
    if shuffle:
        dataset = dataset.shuffle(2000)
    return dataset


'''
get_cross_entropy_loss: Computes softmax cross entropy between logits and labels.
Args:
  labels: Each vector along the class dimension should hold a valid probability
          distribution e.g. for the case in which labels are of shape
          [batch_size, num_classes], each row of labels[i]
          must be a valid probability distribution.
  logits: logits: Per-label activations, typically a linear output.
          These activation energies are interpreted as unnormalized
          log probabilities.
Returns:
  cross_entropy_loss
'''
def get_cross_entropy_loss(labels, logits):
    loss = tf.nn.softmax_cross_entropy_with_logits(labels=labels, logits=logits)
    return tf.reduce_mean(loss)


'''
test_step: gets model prediction for given samples
Args:
  joint_data: tensor joint data
  bone_data : tensor bone data
'''
@tf.function
def test_step(joint_data, bone_data):
    logits = model(joint_data, bone_data, training=False)
    return tf.nn.softmax(logits)

'''
train_step: trains model with cross entropy loss
Args:
  joint_data: tensor joint data
  bone_data : tensor bone data
  labels    : one hot encoded labels
  train_incidence: When true incidence matrices will be trained
'''
@tf.function
def train_step(joint_data, bone_data, labels, train_incidence):
    with tf.GradientTape() as tape:
        logits = model(joint_data, bone_data, training=True)
        loss   = get_cross_entropy_loss(labels=labels, logits=logits)
    trainable_variables = [variable for variable in model.trainable_variables if not "incidence_matrix" in variable.name]
    trainable_variables = model.trainable_variables if train_incidence else trainable_variables
    gradients = tape.gradient(loss, trainable_variables)
    optimizer.apply_gradients(zip(gradients, trainable_variables))
    return loss, tf.nn.softmax(logits)


if __name__ == "__main__":
    parser = get_parser()
    arg = parser.parse_args()

    #copy hyperparameters and model definition to log folder
    save_arg(arg)
    shutil.copy2(inspect.getfile(DGNN), arg.log_dir)

    learning_rate   = arg.base_lr
    num_classes     = arg.num_classes
    batch_size      = arg.batch_size
    epochs          = arg.num_epochs
    checkpoint_path = arg.checkpoint_path
    log_dir         = arg.log_dir
    train_data_path = arg.train_data_path
    test_data_path  = arg.test_data_path
    save_freq       = arg.save_freq
    freeze_graph_until = arg.freeze_graph_until

    '''
    Get tf.dataset objects for training and testing data
    Data shape: bone data  - batch_size, 2, 300, 25, 3
                joint data - batch_size, 2, 300, 25, 3
                labels     - batch_size, num_classes
    '''
    train_data = get_dataset(train_data_path,
                             num_classes=num_classes,
                             batch_size=batch_size,
                             drop_remainder=True,
                             shuffle=True)

    test_data = get_dataset(test_data_path,
                            num_classes=num_classes,
                            batch_size=batch_size,
                            drop_remainder=False,
                            shuffle=False)

    model          = DGNN(num_classes=num_classes)
    optimizer      = tf.keras.optimizers.Adam(learning_rate = learning_rate)
    summary_writer = tf.summary.create_file_writer(log_dir)
    ckpt           = tf.train.Checkpoint(model=model, optimizer=optimizer)
    ckpt_manager   = tf.train.CheckpointManager(ckpt, checkpoint_path, max_to_keep=5)

    # keras metrics to hold accuracies and loss
    cross_entropy_loss = tf.keras.metrics.Mean(name='cross_entropy_loss')
    train_acc          = tf.keras.metrics.CategoricalAccuracy(name='train_acc')
    test_acc           = tf.keras.metrics.CategoricalAccuracy(name='test_acc')
    train_acc_top_k    = tf.keras.metrics.TopKCategoricalAccuracy(name='train_acc_top_k')
    test_acc_top_k     = tf.keras.metrics.TopKCategoricalAccuracy(name='test_acc_top_k')

    # Get 1 batch from train dataset to get graph trace of train and test functions
    for data in train_data:
        joint_data, bone_data, labels = data
        break

    # add graph of train and test functions to tensorboard graphs
    # Note:
    # graph training is True on purpose, allows tensorflow to get all the
    # variables, which is required for the first call of @tf.function function
    tf.summary.trace_on(graph=True)
    train_step(joint_data, bone_data, labels, True)
    with summary_writer.as_default():
      tf.summary.trace_export(name="training_trace",step=0)
    tf.summary.trace_off()

    tf.summary.trace_on(graph=True)
    test_step(joint_data, bone_data)
    with summary_writer.as_default():
      tf.summary.trace_export(name="testing_trace", step=0)
    tf.summary.trace_off()

    # get graph_temporal_conv layers to plot their incidence matrices
    # in order to plot in tensorboard
    gtc_layers = [layer for layer in model.layers if "graph_temporal_conv" in layer.name]

    # start training
    train_iter = 0
    test_iter = 0
    for epoch in range(epochs):
        print("Epoch: {}".format(epoch+1))

        # Using the file writer, log the incidence matrices as images.
        with summary_writer.as_default():
            for layer in gtc_layers:
                tf.summary.image(layer.name+"_incidence_matrix_target",
                                 tf.expand_dims(tf.expand_dims(layer.dgnb.target_A, 0), -1),
                                 step=epoch)
                tf.summary.image(layer.name+"_incidence_matrix_source",
                                 tf.expand_dims(tf.expand_dims(layer.dgnb.source_A, 0), -1),
                                 step=epoch)

        print("Training: ")
        for joint_data, bone_data, labels in tqdm(train_data):
            loss, y_pred = train_step(joint_data, bone_data, labels, True if epoch > freeze_graph_until else False)
            train_acc(labels, y_pred)
            train_acc_top_k(labels, y_pred)
            cross_entropy_loss(loss)
            with summary_writer.as_default():
                tf.summary.scalar("cross_entropy_loss", cross_entropy_loss.result(), step=train_iter)
                tf.summary.scalar("train_acc", train_acc.result(), step=train_iter)
                tf.summary.scalar("train_acc_top_5", train_acc_top_k.result(), step=train_iter)
            cross_entropy_loss.reset_states()
            train_acc.reset_states()
            train_acc_top_k.reset_states()
            train_iter += 1

        print("Testing: ")
        for joint_data, bone_data, labels in tqdm(test_data):
            y_pred = test_step(joint_data, bone_data)
            test_acc(labels, y_pred)
            test_acc_top_k(labels, y_pred)
            with summary_writer.as_default():
                tf.summary.scalar("test_acc", test_acc.result(), step=test_iter)
                tf.summary.scalar("test_acc_top_5", test_acc_top_k.result(), step=test_iter)
            test_acc.reset_states()
            test_acc_top_k.reset_states()
            test_iter += 1

        if (epoch + 1) % save_freq == 0:
            ckpt_save_path = ckpt_manager.save()
            print('Saving checkpoint for epoch {} at {}'.format(epoch+1, ckpt_save_path))

    ckpt_save_path = ckpt_manager.save()
    print('Saving final checkpoint for epoch {} at {}'.format(epochs, ckpt_save_path))

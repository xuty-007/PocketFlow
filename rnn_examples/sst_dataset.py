# Tencent is pleased to support the open source community by making PocketFlow available.
#
# Copyright (C) 2018 THL A29 Limited, a Tencent company. All rights reserved.
#
# Licensed under the BSD 3-Clause License (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://opensource.org/licenses/BSD-3-Clause
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Fashion-MNIST dataset."""

import os
import numpy as np
import pickle
import tensorflow as tf

from datasets.abstract_dataset import AbstractDataset

FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_integer('nb_classes', 2, '# of classes')
tf.app.flags.DEFINE_integer('nb_smpls_train', 6000, '# of samples for training')
tf.app.flags.DEFINE_integer('nb_smpls_val', 920, '# of samples for validation')
tf.app.flags.DEFINE_integer('nb_smpls_eval', 872, '# of samples for evaluation')
tf.app.flags.DEFINE_integer('batch_size', 128, 'batch size per GPU for training')
tf.app.flags.DEFINE_integer('batch_size_eval', 128, 'batch size for evaluation')
tf.app.flags.DEFINE_integer('vocab_size', 13966, 'size of vocabulary')


def parse_fn(x, y, is_train):
  """Parse an (image, label) pair and apply data augmentation if needed.

  Args:
  * image: image tensor
  * label: label tensor
  * is_train: whether data augmentation should be applied

  Returns:
  * image: image tensor
  * label: one-hot label tensor
  """

  # data parsing
  y = tf.one_hot(y, FLAGS.nb_classes)

  return x, y


class SSTDataset(AbstractDataset):
  '''Fashion-MNIST dataset.'''

  def __init__(self, is_train):
    """Constructor function.

    Args:
    * is_train: whether to construct the training subset
    """

    # initialize the base class
    super(SSTDataset, self).__init__(is_train)

    # choose local files or HDFS files w.r.t. FLAGS.data_disk
    if FLAGS.data_disk == 'local':
      assert FLAGS.data_dir_local is not None, '<FLAGS.data_dir_local> must not be None'
      data_dir = FLAGS.data_dir_local
    elif FLAGS.data_disk == 'hdfs':
      assert FLAGS.data_hdfs_host is not None and FLAGS.data_dir_hdfs is not None, \
        'both <FLAGS.data_hdfs_host> and <FLAGS.data_dir_hdfs> must not be None'
      data_dir = FLAGS.data_hdfs_host + FLAGS.data_dir_hdfs
    else:
      raise ValueError('unrecognized data disk: ' + FLAGS.data_disk)

    # setup paths to image & label files, and read in images & labels
    if is_train:
      self.batch_size = FLAGS.batch_size
      filename = os.path.join(data_dir, 'SST_train.pickle')
    else:
      self.batch_size = FLAGS.batch_size_eval
      filename = os.path.join(data_dir, 'SST_test.pickle')
    self.x, self.y = pickle.load(open(filename, 'rb'))
    self.parse_fn = lambda x, y: parse_fn(x, y, is_train)

  def build(self, enbl_trn_val_split=False):
    """Build iterator(s) for tf.data.Dataset() object.

    Args:
    * enbl_trn_val_split: whether to split into training & validation subsets

    Returns:
    * iterator_trn: iterator for the training subset
    * iterator_val: iterator for the validation subset
      OR
    * iterator: iterator for the chosen subset (training OR testing)
    """

    # create a tf.data.Dataset() object from NumPy arrays
    x = tf.convert_to_tensor(self.x)
    y = tf.convert_to_tensor(self.y)
    dataset = tf.data.Dataset.from_tensor_slices((x, y))
    dataset = dataset.map(self.parse_fn, num_parallel_calls=FLAGS.nb_threads)

    # create iterators for training & validation subsets separately
    if self.is_train and enbl_trn_val_split:
      iterator_val = self.__make_iterator(dataset.take(FLAGS.nb_smpls_val))
      iterator_trn = self.__make_iterator(dataset.skip(FLAGS.nb_smpls_val))
      return iterator_trn, iterator_val

    return self.__make_iterator(dataset)

  def __make_iterator(self, dataset):
    """Make an iterator from tf.data.Dataset.

    Args:
    * dataset: tf.data.Dataset object

    Returns:
    * iterator: iterator for the dataset
    """

    dataset = dataset.apply(tf.contrib.data.shuffle_and_repeat(buffer_size=FLAGS.buffer_size))
    dataset = dataset.batch(self.batch_size)
    dataset = dataset.prefetch(FLAGS.prefetch_size)
    iterator = dataset.make_one_shot_iterator()

    return iterator

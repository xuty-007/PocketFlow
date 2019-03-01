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
"""Utility functions."""

import os
import subprocess
import tensorflow as tf
from tensorflow.contrib.quantize.python import common
from tensorflow.contrib.quantize.python import input_to_ops
from tensorflow.contrib.quantize.python import quant_ops

from utils.misc_utils import auto_barrier
from utils.misc_utils import is_primary_worker
from utils.multi_gpu_wrapper import MultiGpuWrapper as mgw

FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_string('skrnn_save_path_probe', './models_skrnn_probe/model.ckpt',
                           'Skip-RNN: probe model\'s save path')
tf.app.flags.DEFINE_string('skrnn_save_path_probe_eval', './models_skrnn_probe_eval/model.ckpt',
                           'Skip-RNN: probe model\'s save path for evaluation')

def create_session():
  """Create a TensorFlow session.

  Return:
  * sess: TensorFlow session
  """

  # create a TensorFlow session
  config = tf.ConfigProto()
  config.gpu_options.visible_device_list = str(mgw.local_rank() if FLAGS.enbl_multi_gpu else 0)  # pylint: disable=no-member
  config.gpu_options.allow_growth = True  # pylint: disable=no-member
  sess = tf.Session(config=config)

  return sess


class SkipRNN:
  # pylint: disable=too-many-instance-attributes
  """ Class of uniform quantization """
  allowed_rnn_cells = {"BasicLSTMCell": "basic_lstm_cell"}

  def __init__(self, sess, cost_per_sample=0.0, rnn_cell="BasicLSTMCell"):
    self.sess = sess
    self.cost_per_sample = cost_per_sample
    self.rnn_cell = rnn_cell
    self.input_ops = []
    self.output_ops = []
    self.__safe_check()

  def search_init_inputs(self):
    self.

  def search_inputs(self):
    """ search matmul or Conv2D operations in graph for quantization"""

    is_student_fn = lambda x: 'distilled' not in x.name
    self.input_ops = [ops for ops in self.sess.graph.get_operations()
                      if is_student_fn(ops)
                      and self.allowed_rnn_cells[self.rnn_cell] in ops.name
                      and any([self.allowed_rnn_cells[self.rnn_cell] not in v.name
                               for v in ops.inputs])]

    self.inputs = {}
    if self.rnn_cell == "BasicLSTMCell":
      for op in self.input_ops:
        if "concat" in op.name:
          for v in op.inputs:
            if "TensorArrayReadV3" in v.name:
              self.inputs["x"] = v
            elif "Identity" in v.name:
              self.inputs["h"] = v
        if "Mul" in op.name:
          for v in op.name:
            if "Identity" in v.name:
              self.inputs["c"] = v
    return self.inputs

  def build(self):
    x = self.inputs["x"]
    h = self.inputs["h"]
    c = self.inputs["c"]
  def __scale(self, w, mode):
    """linear scale function

    Args:
    * w: A Tensor (weights or activation output),
         the shape is [bucket_size, bucekt_num] if use_buckets else the original size.
    * mode: A string, 'weight' or 'activation'

    Returns:
    * A Tensor, the normalized weights
    * A Tensor, alpha, scalar if activation mode else a vector [bucket_num].
    * A Tensor, beta, scalar if activation mode else a vector [bucket_num].
    """
    if mode == 'weight':
      if self.use_buckets:
        axis = 0
      else:
        axis = None
    elif mode == 'activation':
      axis = None
    else:
      raise ValueError("Unknown mode for scalling")

    w_max = tf.stop_gradient(tf.reduce_max(w, axis=axis))
    w_min = tf.stop_gradient(tf.reduce_min(w, axis=axis))
    eps = tf.constant(value=1e-10, dtype=tf.float32)

    alpha = w_max - w_min + eps
    beta = w_min
    w = (w - beta) / alpha
    return w, alpha, beta

  def __inv_scale(self, w, alpha, beta):
    """Inversed linear scale function

    Args:
    * w: A Tensor (weights or activation output)
    * alpha: A float value, scale factor
    * bete: A float value, scale bias

    Returns:
    * A Tensor, inversed scale value1
    """

    return alpha * w + beta

  def __split_bucket(self, w):
    """Create bucket

    Args:
    * w: A Tensor (weights)

    Returns:
    * A Tensor, with shape [bucket_size, multiple]
    * An integer: the number of buckets
    * An integer, the number of padded elements
    """

    flat_w = tf.reshape(w, [-1])
    num_w = flat_w.get_shape()[0].value
    # use the last value to fill
    fill_value = flat_w[-1]

    multiple, rest = divmod(num_w, self.bucket_size)
    if rest != 0:
      values_to_add = tf.ones(self.bucket_size - rest) * fill_value
      # add the fill_value to make the tensor into a multiple of the bucket size.
      flat_w = tf.concat([flat_w, values_to_add], axis=0)
      multiple += 1

    flat_w = tf.reshape(flat_w, [self.bucket_size, -1])
    padded_num = (self.bucket_size - rest) if rest != 0 else 0

    return flat_w, multiple, padded_num

  def __channel_bucket(self, w):
    """ reshape weights according to bucket for 'channel' type.
        Note that for fc layers, buckets are created row-wisely.
    Args:
      w: A Tensor (weights)

    Returns:
      A Tensor shape [bucket_size, bucket_num], bucket_size = h*w*cin for conv or cin for fc
      A integer: the number of buckets
      A integer (0), zero padded elements
    """
    cout = w.get_shape()[-1].value
    folded_w = tf.reshape(w, [-1, cout])
    return folded_w, cout, 0

  def __safe_check(self):
    """ TODO: Check the name of bucket_type, the value of bucket_size """

    if self.cost_per_sample < 0:
      raise ValueError("Cost per sample must be a postive float.")

  def __updt_bucket_storage(self, bucket_num):
    """ Calculate extra storage for the bucket scalling factors

    Args:
    * bucket_num: a Tensor, the number of buckets, and 2*bucket_num scalling factors
    * alpha: a Tensor, the scalling factor
    """
    self.bucket_storage += bucket_num * 32 * 2  # both alpha and beta, so *2




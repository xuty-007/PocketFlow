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
import collections
import tensorflow as tf
from tensorflow.python.ops.rnn_cell import LayerRNNCell, RNNCell
from tensorflow.python.ops import init_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import nn_ops
from tensorflow.python.framework import ops

from utils.misc_utils import auto_barrier
from utils.misc_utils import is_primary_worker
from utils.multi_gpu_wrapper import MultiGpuWrapper as mgw

_BIAS_VARIABLE_NAME = "bias"
_WEIGHTS_VARIABLE_NAME = "kernel"

SkipLSTMStateTuple = collections.namedtuple("SkipLSTMStateTuple", ("c", "h", "update_prob", "cum_update_prob"))
SkipLSTMOutputTuple = collections.namedtuple("SkipLSTMOutputTuple", ("h", "state_gate"))
LSTMStateTuple = tf.nn.rnn_cell.LSTMStateTuple
GRUStateTuple = lambda x: x

SkipGRUStateTuple = collections.namedtuple("SkipGRUStateTuple", ("h", "update_prob", "cum_update_prob"))
SkipGRUOutputTuple = collections.namedtuple("SkipGRUOutputTuple", ("h", "state_gate"))

FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_string('skrnn_save_path_probe', './models_skrnn_probe/model.ckpt',
                           'Skip-RNN: probe model\'s save path')
tf.app.flags.DEFINE_string('skrnn_save_path_probe_eval', './models_skrnn_probe_eval/model.ckpt',
                           'Skip-RNN: probe model\'s save path for evaluation')


def _binary_round(x):
  """
  Rounds a tensor whose values are in [0,1] to a tensor with values in {0, 1},
  using the straight through estimator for the gradient.

  Based on http://r2rt.com/binary-stochastic-neurons-in-tensorflow.html

  :param x: input tensor
  :return: y=round(x) with gradients defined by the identity mapping (y=x)
  """
  g = tf.get_default_graph()

  with ops.name_scope("BinaryRound") as name:
    with g.gradient_override_map({"Round": "Identity"}):
      return math_ops.round(x, name=name)


class SkipRNNCell(LayerRNNCell):
  # pylint: disable=too-many-instance-attributes
  """ Class of uniform quantization """
  allowed_rnn_cells = [tf.nn.rnn_cell.BasicLSTMCell, tf.nn.rnn_cell.GRUCell]

  def __init__(self,
               cell,
               name=None,
               update_bias=1.0):
    if name is None:
      name = cell._name
    super(SkipRNNCell, self).__init__(
      _reuse=cell._reuse, name=name)

    if any([isinstance(cell, t) for t in self.allowed_rnn_cells]):
      self.cell = cell
      self._cell_type = [isinstance(cell, t) for t in self.allowed_rnn_cells].index(True)

    else:
      raise NotImplementedError("Unspport RNN cell: %s." % type(cell))
    self._num_units = cell._num_units
    self._activation = cell._activation
    self._update_bias = update_bias
    self._forget_bias = cell._forget_bias
    self.input_ops = []
    self.output_ops = []

  @property
  def state_size(self):
    if self._cell_type == 0:
      return SkipLSTMStateTuple(self._num_units, self._num_units, 1, 1)
    if self._cell_type == 1:
      return SkipGRUStateTuple(self._num_units, 1, 1)

  @property
  def output_size(self):
    if self._cell_type == 0:
      return SkipLSTMOutputTuple(self._num_units, 1)
    if self._cell_type == 1:
      return SkipGRUOutputTuple(self._num_units, 1)

  def zero_state(self, batch_size, dtype):
    state = super(SkipRNNCell, self).zero_state(batch_size, dtype)
    state[-1] = tf.get_variable("initial_update_prob", shape=[batch_size, 1], trainable=False,
                                  initializer=tf.ones_initializer())
    state[-2] = tf.get_variable("initial_cum_update_prob", shape=[batch_size, 1],
                                      trainable=False, initializer=tf.zeros_initializer())
    return state

  def build(self, inputs_shape):
    if inputs_shape[1].value is None:
      raise ValueError("Expected inputs.shape[-1] to be known, saw shape: %s"
                       % inputs_shape)

    input_depth = inputs_shape[1].value
    h_depth = self._num_units
    self._kernel = self.add_variable(
        _WEIGHTS_VARIABLE_NAME,
        shape=[h_depth, 1])
    self._bias = self.add_variable(
        _BIAS_VARIABLE_NAME,
        shape=[1],
        initializer=init_ops.constant_initializer(self._update_bias, dtype=self.dtype))

    self.built = True

  def call(self, inputs, state):
    if self._cell_type == 0:
      return self._call_BasicLSTMCell(inputs, state)
    elif self._cell_type == 1:
      return self._call_GRUCell(inputs, state)

  def _call_BasicLSTMCell(self, inputs, state):
    c_prev, h_prev, update_prob_prev, cum_update_prob_prev = state
    _, state_prev = self.cell(inputs, LSTMStateTuple(c_prev, h_prev))
    new_c_tilde, new_h_tilde = state_prev

    new_update_prob, new_cum_update_prob, update_gate = self._skip_RNNCell(
      c_prev, cum_update_prob_prev,
      update_prob_prev)

    new_c = update_gate * new_c_tilde + (1. - update_gate) * c_prev
    new_h = update_gate * new_h_tilde + (1. - update_gate) * h_prev
    new_state = SkipLSTMStateTuple(new_c, new_h, new_update_prob, new_cum_update_prob)
    new_output = SkipLSTMOutputTuple(new_h, update_gate)

    return new_output, new_state

  def _call_GRUCell(self, inputs, state):
    h_prev, update_prob_prev, cum_update_prob_prev = state
    _, state_prev = self.cell(inputs, h_prev)
    new_h_tilde = state_prev

    new_update_prob, new_cum_update_prob, update_gate = self._skip_RNNCell(
      h_prev, cum_update_prob_prev,
      update_prob_prev)

    new_h = update_gate * new_h_tilde + (1. - update_gate) * h_prev
    new_state = SkipGRUStateTuple(new_h, new_update_prob, new_cum_update_prob)
    new_output = SkipGRUOutputTuple(new_h, update_gate)

    return new_output, new_state

  def _skip_RNNCell(self, p_prev, cum_update_prob_prev, update_prob_prev):
    with tf.variable_scope('state_update_prob'):
      new_update_prob_tilde = math_ops.matmul(p_prev, self._kernel)
      new_update_prob_tilde = nn_ops.bias_add(new_update_prob_tilde, self._bias)
      new_update_prob_tilde = math_ops.sigmoid(new_update_prob_tilde)

    # Compute value for the update gate
    cum_update_prob = cum_update_prob_prev + math_ops.minimum(update_prob_prev,
                                                              1. - cum_update_prob_prev)
    update_gate = _binary_round(cum_update_prob)

    # Apply update gate
    new_update_prob = update_gate * new_update_prob_tilde + (1. - update_gate) * update_prob_prev
    new_cum_update_prob = update_gate * 0. + (1. - update_gate) * cum_update_prob

    return new_update_prob, new_cum_update_prob, update_gate

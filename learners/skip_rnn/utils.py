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
from tensorflow.python.ops.rnn_cell_impl import LayerRNNCell, RNNCell
from tensorflow.python.ops import init_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import variable_scope as vs
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


def warm_start(trainable_vars, model_scope, sess):
  """Initialize the model for warm-start.

  Description:
  * We use a pre-trained ImageNet classification model to initialize the backbone part of the SSD
    model for feature extraction. If the SSD model's checkpoint files already exist, then the
    learner should restore model weights by itself.
  """

  # # obtain a list of scopes to be excluded from initialization
  # excl_scopes = []
  # if FLAGS.warm_start_excl_scopes:
  #   excl_scopes = [scope.strip() for scope in FLAGS.warm_start_excl_scopes.split(',')]
  # tf.logging.info('excluded scopes: {}'.format(excl_scopes))
  #
  # # obtain a list of variables to be initialized
  # vars_list = []
  # for var in self.trainable_vars:
  #   excluded = False
  #   for scope in excl_scopes:
  #     if scope in var.name:
  #       excluded = True
  #       break
  #   if not excluded:
  #     vars_list.append(var)
  #
  # rename the variables' scope
  vars_list = {}
  if FLAGS.backbone_model_scope is not None:
    backbone_model_scope = FLAGS.backbone_model_scope.strip()
    if backbone_model_scope == '':
      vars_list = {var.op.name.replace(model_scope + '/', ''): var for var in trainable_vars}
    else:
      vars_list = {var.op.name.replace(
        model_scope, backbone_model_scope): var for var in trainable_vars}
  #
  # # re-map the variables' names
  # name_remap = {'/kernel': '/weights', '/bias': '/biases'}
  # vars_list_remap = {}
  # for var_name, var in vars_list.items():
  #   for name_old, name_new in name_remap.items():
  #     if name_old in var_name:
  #       var_name = var_name.replace(name_old, name_new)
  #       break
  #   vars_list_remap[var_name] = var
  # vars_list = vars_list_remap

  # # display all the variables to be initialized
  # for var_name, var in vars_list.items():
  #   tf.logging.info('using %s to initialize %s' % (var_name, var.op.name))
  # if not vars_list:
  #   raise ValueError('variables to be restored cannot be empty')

  # obtain the checkpoint files' path
  ckpt_path = tf.train.latest_checkpoint(FLAGS.backbone_ckpt_dir)
  tf.logging.info('restoring model weights from ' + ckpt_path)

  # remove missing variables from the list
  if FLAGS.ignore_missing_vars:
    reader = tf.train.NewCheckpointReader(ckpt_path)
    vars_list_avail = {}
    for var,value in vars_list.items():
      if reader.has_tensor(var):
        vars_list_avail[var] = value
      else:
        tf.logging.warning('variable %s not found in checkpoint files %s.' % (var, ckpt_path))
    vars_list = vars_list_avail
  if not vars_list:
    tf.logging.warning('no variables to restore.')
    return

  # restore variables from checkpoint files
  saver = tf.train.Saver(vars_list, reshape=False)
  saver.build()
  saver.restore(sess, ckpt_path)

class SkipRNNCell(LayerRNNCell):
  # pylint: disable=too-many-instance-attributes
  """ Class of uniform quantization """
  allowed_rnn_cells = [tf.nn.rnn_cell.BasicLSTMCell,
                       tf.nn.rnn_cell.GRUCell]

  allowed_multi_rnn_cells = [tf.nn.rnn_cell.MultiRNNCell]

  def __init__(self,
               cell,
               name=None,
               update_bias=1.0):
    if name is None:
      name = cell._name
    super(SkipRNNCell, self).__init__(
      _reuse=cell._reuse, name=name)

    if any([isinstance(cell, t) for t in self.allowed_rnn_cells + self.allowed_multi_rnn_cells]):
      self.cell = cell
      if any([isinstance(cell, t) for t in self.allowed_multi_rnn_cells]):
        self.is_multi_rnn = True
        c = cell._cells[-1]
      else:
        self.is_multi_rnn = False
        c = cell
      _cell_type = [isinstance(c, t) for t in self.allowed_rnn_cells].index(True)
    else:
      raise NotImplementedError("Unspport RNN cell: %s." % type(cell))

    if self.is_multi_rnn:
      tf.logging.info("Running for %s of %s ..." % (type(cell), type(cell._cells[0])))
    else:
      tf.logging.info("Running for %s ..." % type(cell))
    if _cell_type == 0:
      self.SkipStateTuple = SkipLSTMStateTuple
      self.SkipOutputTuple = SkipLSTMOutputTuple
      self.Get_State = lambda state: LSTMStateTuple(state.c, state.h)
      self.Prob = lambda state: state.c
      self.Get_Output = lambda state: state.h
      self.toStateTuple = lambda state: LSTMStateTuple(*state)
      # self.call = self._call_BasicLSTMCell
    elif _cell_type == 1:
      self.SkipStateTuple = SkipGRUStateTuple
      self.SkipOutputTuple = SkipGRUOutputTuple
      self.Get_State = lambda state: state.h
      self.Prob = lambda state: state.h
      self.Get_Output = lambda state: state.h
      self.toStateTuple = lambda state: state[0]
      # self.call = self._call_GRUCell

    if self.is_multi_rnn:
      self._num_units = cell._cells[-1]._num_units
      self.call = self._call_MultiRNNCell
    else:
      self._num_units = cell._num_units
      self.call = self._call_RNNCell
    self._update_bias = update_bias

  @property
  def state_size(self):
    rnn_stat_size = self.cell.state_size

    if self.is_multi_rnn:
      _num_units = rnn_stat_size[-1]
    else:
      _num_units = rnn_stat_size

    if isinstance(_num_units, (tuple, list)):
      state_size = self.SkipStateTuple(*_num_units, update_prob=1, cum_update_prob=1)
    else:
      state_size = self.SkipStateTuple(_num_units, update_prob=1, cum_update_prob=1)

    if self.is_multi_rnn:
      return tuple(list(rnn_stat_size[:-1]) + [state_size])

    return state_size

  @property
  def output_size(self):
    _num_units = self.cell.output_size
    if isinstance(_num_units, (tuple,list)):
      return self.SkipOutputTuple(*_num_units, state_gate=1)
    else:
      return self.SkipOutputTuple(_num_units, state_gate=1)

  def zero_state(self, batch_size, dtype):
    state = super(SkipRNNCell, self).zero_state(batch_size, dtype)
    with ops.name_scope(type(self).__name__ + "ZeroState", values=[batch_size]):
      update_prob = array_ops.ones([batch_size, 1], dtype=dtype)
      cum_update_prob = array_ops.zeros([batch_size, 1], dtype=dtype)

    if self.is_multi_rnn:
      skip_zero_state = self.SkipStateTuple(*state[-1][:-2],
                                            update_prob=update_prob,
                                            cum_update_prob=cum_update_prob)
      return tuple(list(state[:-1]) + [skip_zero_state])

    return self.SkipStateTuple(*state[:-2],
                                update_prob=update_prob,
                                cum_update_prob=cum_update_prob)

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

  def _call_MultiRNNCell(self, inputs, state):
    update_prob_prev, cum_update_prob_prev = state[-1].update_prob, state[-1].cum_update_prob
    states_prev = tuple(list(state[:-1]) + [self.Get_State(state[-1])])
    _, state_candidates = self.cell(inputs, states_prev)

    new_update_prob, new_cum_update_prob, update_gate = self._skip_RNNCell(
      self.Prob(state[-1]), cum_update_prob_prev, update_prob_prev
    )

    # Update statest
    new_states = []
    new_state = []
    for state_candidate, state_prev in zip(state_candidates, states_prev):
      new_state = []
      if isinstance(state_candidate, (tuple, list)):
        for new_v_tilde, v_prev in zip(state_candidate, state_prev):
          new_state.append(update_gate * new_v_tilde + (1. - update_gate) * v_prev)
      else:
        new_state.append(update_gate * state_candidate + (1. - update_gate) * state_prev)
      new_states.append(self.toStateTuple(new_state))

    new_states[-1] = self.SkipStateTuple(*new_state, update_prob=new_update_prob,
                                            cum_update_prob=new_cum_update_prob)
    new_outputs = self.SkipOutputTuple(self.Get_Output(new_states[-1]), update_gate)

    return new_outputs, tuple(new_states)

  def _call_RNNCell(self, inputs, state):
    update_prob_prev, cum_update_prob_prev = state.update_prob, state.cum_update_prob
    state_prev = self.Get_State(state)
    _, new_state_tilde = self.cell(inputs, state_prev)

    new_update_prob, new_cum_update_prob, update_gate = self._skip_RNNCell(
      self.Prob(state), cum_update_prob_prev, update_prob_prev)

    new_state = []
    if isinstance(new_state_tilde, (tuple, list)):
      for new_v_tilde, v_prev in zip(new_state_tilde, state_prev):
        new_state.append(update_gate * new_v_tilde + (1. - update_gate) * v_prev)
    else:
      new_state.append(update_gate * new_state_tilde + (1. - update_gate) * state_prev)

    new_state = self.SkipStateTuple(*new_state, update_prob=new_update_prob,
                                    cum_update_prob=new_cum_update_prob)
    new_output = self.SkipOutputTuple(self.Get_Output(new_state), update_gate)

    return new_output, new_state

  def _call_BasicLSTMCell(self, inputs, state):
    c_prev, h_prev, update_prob_prev, cum_update_prob_prev = state
    _, state_prev = self.cell(inputs, LSTMStateTuple(c_prev, h_prev))
    new_c_tilde, new_h_tilde = state_prev

    new_update_prob, new_cum_update_prob, update_gate = self._skip_RNNCell(
      c_prev, cum_update_prob_prev, update_prob_prev)

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
      h_prev, cum_update_prob_prev, update_prob_prev)

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

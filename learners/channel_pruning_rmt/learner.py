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
"""Channel pruning learner - remastered."""

import os
import re
import math
from timeit import default_timer as timer
import numpy as np
from scipy.linalg import norm
import tensorflow as tf

from learners.abstract_learner import AbstractLearner
from learners.distillation_helper import DistillationHelper
from utils.multi_gpu_wrapper import MultiGpuWrapper as mgw

FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_string('cpr_save_path', './models_cpr/model.ckpt', 'CPR: model\'s save path')
tf.app.flags.DEFINE_string('cpr_save_path_eval', './models_cpr_eval/model.ckpt',
                           'CPR: model\'s save path for evaluation')
tf.app.flags.DEFINE_float('cpr_prune_ratio', 0.5, 'CPR: pruning ratio')
tf.app.flags.DEFINE_boolean('cpr_skip_frst_layer', True, 'CPR: skip the first layer for pruning')
tf.app.flags.DEFINE_boolean('cpr_skip_last_layer', False, 'CPR: skip the last layer for pruning')
tf.app.flags.DEFINE_integer('cpr_nb_smpl_insts', 5000, 'CPR: # of sampled training instances')
tf.app.flags.DEFINE_integer('cpr_nb_smpl_crops', 10, 'CPR: # of sampled random crops per instance')
tf.app.flags.DEFINE_float('cpr_ista_lrn_rate', 1e-2, 'CPR: ISTA\'s learning rate')
tf.app.flags.DEFINE_integer('cpr_ista_nb_iters', 100, 'CPR: # of iterations in ISTA')
tf.app.flags.DEFINE_boolean('cpr_eval_per_layer', False, 'CPR: evaluate whenever a layer is pruned')

def get_vars_by_scope(scope):
  """Get list of variables within certain name scope.

  Args:
  * scope: name scope

  Returns:
  * vars_dict: dictionary of list of all, trainable, and convolutional kernel variables
  """

  vars_dict = {}
  vars_dict['all'] = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=scope)
  vars_dict['trainable'] = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=scope)
  vars_dict['conv_krnl'] = []
  conv2d_pattern = re.compile(r'/Conv2D$')
  conv2d_ops = get_ops_by_scope_n_pattern(scope, conv2d_pattern)
  for var in vars_dict['trainable']:
    for op in conv2d_ops:
      for op_input in op.inputs:
        if op_input.name == var.name.replace(':0', '/read:0'):
          vars_dict['conv_krnl'] += [var]
          break

  return vars_dict

def get_ops_by_scope_n_pattern(scope, pattern):
  """Get list of operations within certain name scope and also matches the pattern.

  Args:
  * scope: name scope
  * pattern: name pattern to be matched

  Returns:
  * ops: list of operations
  """

  ops = []
  for op in tf.get_default_graph().get_operations():
    if op.name.startswith(scope) and re.search(pattern, op.name) is not None:
      ops += [op]

  return ops

def calc_prune_ratio(vars_list):
  """Calculate the overall pruning ratio for the given list of variables.

  Args:
  * vars_list: list of variables

  Returns:
  * prune_ratio: overall pruning ratio of the given list of variables
  """

  nb_params_nnz = tf.add_n([tf.count_nonzero(var) for var in vars_list])
  nb_params_all = tf.add_n([tf.size(var) for var in vars_list])
  prune_ratio = 1.0 - tf.cast(nb_params_nnz, tf.float32) / tf.cast(nb_params_all, tf.float32)

  return prune_ratio

class ChannelPrunedRmtLearner(AbstractLearner):  # pylint: disable=too-many-instance-attributes
  """Channel pruning learner - remastered."""

  def __init__(self, sm_writer, model_helper):
    """Constructor function.

    Args:
    * sm_writer: TensorFlow's summary writer
    * model_helper: model helper with definitions of model & dataset
    """

    # class-independent initialization
    super(ChannelPrunedRmtLearner, self).__init__(sm_writer, model_helper)

    # define scopes for full & channel-pruned models
    self.model_scope_full = 'model'
    self.model_scope_prnd = 'pruned_model'

    # download the pre-trained model
    if self.is_primary_worker('local'):
      self.download_model()  # pre-trained model is required
    self.auto_barrier()
    tf.logging.info('model files: ' + ', '.join(os.listdir('./models')))

    # class-dependent initialization
    if FLAGS.enbl_dst:
      self.helper_dst = DistillationHelper(sm_writer, model_helper, self.mpi_comm)
    self.__build_train()
    self.__build_eval()

  def train(self):
    """Train a model and periodically produce checkpoint files."""

    # restore the full model from pre-trained checkpoints
    save_path = tf.train.latest_checkpoint(os.path.dirname(self.save_path_full))
    self.saver_full.restore(self.sess_train, save_path)

    # initialization
    self.sess_train.run(self.init_op)
    if FLAGS.enbl_multi_gpu:
      self.sess_train.run(self.bcast_op)

    # choose channels and evaluate the model before re-training
    time_prev = timer()
    self.__choose_channels()
    tf.logging.info('time (channel selection): %.2f (s)' % (timer() - time_prev))
    self.sess_train.run(self.mask_updt_op)
    if FLAGS.enbl_multi_gpu:
      self.sess_train.run(self.bcast_op)

    # evaluate the model before fine-tuning
    if self.is_primary_worker('global'):
      self.__save_model(is_train=True)
      self.evaluate()
    self.auto_barrier()

    # fine-tune the model with chosen channels only
    time_prev = timer()
    for idx_iter in range(self.nb_iters_train):
      # train the model
      if (idx_iter + 1) % FLAGS.summ_step != 0:
        self.sess_train.run(self.train_op)
      else:
        __, summary, log_rslt = self.sess_train.run([self.train_op, self.summary_op, self.log_op])
        if self.is_primary_worker('global'):
          time_step = timer() - time_prev
          self.__monitor_progress(summary, log_rslt, idx_iter, time_step)
          time_prev = timer()

      # save the model at certain steps
      if self.is_primary_worker('global') and (idx_iter + 1) % FLAGS.save_step == 0:
        self.__save_model(is_train=True)
        self.evaluate()
      self.auto_barrier()

    # save the final model
    if self.is_primary_worker('global'):
      self.__save_model(is_train=True)
      self.__restore_model(is_train=False)
      self.__save_model(is_train=False)
      self.evaluate()

  def evaluate(self):
    """Restore a model from the latest checkpoint files and then evaluate it."""

    self.__restore_model(is_train=False)
    nb_iters = int(np.ceil(float(FLAGS.nb_smpls_eval) / FLAGS.batch_size_eval))
    eval_rslts = np.zeros((nb_iters, len(self.eval_op)))
    self.dump_n_eval(outputs=None, action='init')
    for idx_iter in range(nb_iters):
      if (idx_iter + 1) % 100 == 0:
        tf.logging.info('process the %d-th mini-batch for evaluation' % (idx_iter + 1))
      eval_rslts[idx_iter], outputs = self.sess_eval.run([self.eval_op, self.outputs_eval])
      self.dump_n_eval(outputs=outputs, action='dump')
    self.dump_n_eval(outputs=None, action='eval')
    for idx, name in enumerate(self.eval_op_names):
      tf.logging.info('%s = %.4e' % (name, np.mean(eval_rslts[:, idx])))

  def __build_train(self):  # pylint: disable=too-many-locals,too-many-statements
    """Build the training graph."""

    with tf.Graph().as_default():
      # create a TF session for the current graph
      config = tf.ConfigProto()
      config.gpu_options.allow_growth = True  # pylint: disable=no-member
      config.gpu_options.visible_device_list = \
        str(mgw.local_rank() if FLAGS.enbl_multi_gpu else 0)  # pylint: disable=no-member
      sess = tf.Session(config=config)

      # data input pipeline
      with tf.variable_scope(self.data_scope):
        iterator = self.build_dataset_train()
        images, labels = iterator.get_next()

      # model definition - distilled model
      if FLAGS.enbl_dst:
        logits_dst = self.helper_dst.calc_logits(sess, images)

      # model definition - full model
      with tf.variable_scope(self.model_scope_full):
        __ = self.forward_train(images)
        self.vars_full = get_vars_by_scope(self.model_scope_full)
        self.saver_full = tf.train.Saver(self.vars_full['all'])
        self.save_path_full = FLAGS.save_path

      # model definition - channel-pruned model
      with tf.variable_scope(self.model_scope_prnd):
        logits_prnd = self.forward_train(images)
        self.vars_prnd = get_vars_by_scope(self.model_scope_prnd)
        self.conv_krnl_var_names = [var.name for var in self.vars_prnd['conv_krnl']]
        self.global_step = tf.train.get_or_create_global_step()
        self.saver_prnd_train = tf.train.Saver(self.vars_prnd['all'] + [self.global_step])

        # loss & extra evaluation metrics
        loss, metrics = self.calc_loss(labels, logits_prnd, self.vars_prnd['trainable'])
        if FLAGS.enbl_dst:
          loss += self.helper_dst.calc_loss(logits_prnd, logits_dst)
        tf.summary.scalar('loss', loss)
        for key, value in metrics.items():
          tf.summary.scalar(key, value)

        # learning rate schedule
        lrn_rate, self.nb_iters_train = self.setup_lrn_rate(self.global_step)

        # calculate pruning ratios
        pr_trainable = calc_prune_ratio(self.vars_prnd['trainable'])
        pr_conv_krnl = calc_prune_ratio(self.vars_prnd['conv_krnl'])
        tf.summary.scalar('pr_trainable', pr_trainable)
        tf.summary.scalar('pr_conv_krnl', pr_conv_krnl)

        # create masks and corresponding operations for channel pruning
        self.masks = []
        mask_updt_ops = []  # update the mask based on convolutional kernel's value
        for idx, var in enumerate(self.vars_prnd['conv_krnl']):
          tf.logging.info('creating a pruning mask for {} of size {}'.format(var.name, var.shape))
          mask_name = '/'.join(var.name.split('/')[1:]).replace(':0', '_mask')
          mask_shape = [1, 1, var.shape[2], 1]  # 1 x 1 x c_in x 1
          mask = tf.get_variable(mask_name, initializer=tf.ones(mask_shape), trainable=False)
          var_norm = tf.reduce_sum(tf.square(var), axis=[0, 1, 3], keepdims=True)
          self.masks += [mask]
          mask_updt_ops += [mask.assign(tf.cast(var_norm > 0.0, tf.float32))]
        self.mask_updt_op = tf.group(mask_updt_ops)

        # build operations for channel selection
        self.__build_chn_select_ops()

        # optimizer & gradients
        optimizer_base = tf.train.MomentumOptimizer(lrn_rate, FLAGS.momentum)
        if not FLAGS.enbl_multi_gpu:
          optimizer = optimizer_base
        else:
          optimizer = mgw.DistributedOptimizer(optimizer_base)
        grads_origin = optimizer.compute_gradients(loss, self.vars_prnd['trainable'])
        grads_pruned = self.__calc_grads_pruned(grads_origin)
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS, scope=self.model_scope_prnd)
        with tf.control_dependencies(update_ops):
          self.train_op = optimizer.apply_gradients(grads_pruned, global_step=self.global_step)

        # TF operations for initializing the channel-pruned model
        init_ops = []
        for var_full, var_prnd in zip(self.vars_full['all'], self.vars_prnd['all']):
          init_ops += [var_prnd.assign(var_full)]
        init_ops += [self.global_step.initializer]  # initialize the global step
        init_ops += [tf.variables_initializer(optimizer_base.variables())]
        self.init_op = tf.group(init_ops)

      # TF operations for logging & summarizing
      self.sess_train = sess
      self.summary_op = tf.summary.merge_all()
      self.log_op = [lrn_rate, loss, pr_trainable, pr_conv_krnl] + list(metrics.values())
      self.log_op_names = ['lr', 'loss', 'pr_trn', 'pr_krn'] + list(metrics.keys())
      if FLAGS.enbl_multi_gpu:
        self.bcast_op = mgw.broadcast_global_variables(0)

  def __build_eval(self):
    """Build the evaluation graph."""

    with tf.Graph().as_default():
      # create a TF session for the current graph
      config = tf.ConfigProto()
      config.gpu_options.allow_growth = True  # pylint: disable=no-member
      config.gpu_options.visible_device_list = \
        str(mgw.local_rank() if FLAGS.enbl_multi_gpu else 0)  # pylint: disable=no-member
      self.sess_eval = tf.Session(config=config)

      # data input pipeline
      with tf.variable_scope(self.data_scope):
        iterator = self.build_dataset_eval()
        images, labels = iterator.get_next()

      # model definition - distilled model
      if FLAGS.enbl_dst:
        logits_dst = self.helper_dst.calc_logits(self.sess_eval, images)

      # model definition - channel-pruned model
      with tf.variable_scope(self.model_scope_prnd):
        logits = self.forward_eval(images)
        vars_prnd = get_vars_by_scope(self.model_scope_prnd)
        global_step = tf.train.get_or_create_global_step()
        self.saver_prnd_eval = tf.train.Saver(vars_prnd['all'] + [global_step])

        # loss & extra evaluation metrics
        loss, metrics = self.calc_loss(labels, logits, vars_prnd['trainable'])
        if FLAGS.enbl_dst:
          loss += self.helper_dst.calc_loss(logits, logits_dst)

        # calculate pruning ratios
        pr_trainable = calc_prune_ratio(vars_prnd['trainable'])
        pr_conv_krnl = calc_prune_ratio(vars_prnd['conv_krnl'])

        # TF operations for evaluation
        self.eval_op = [loss, pr_trainable, pr_conv_krnl] + list(metrics.values())
        self.eval_op_names = ['loss', 'pr_trn', 'pr_krn'] + list(metrics.keys())
        self.outputs_eval = logits

      # add input & output tensors to certain collections
      tf.add_to_collection('images_final', images)
      tf.add_to_collection('logits_final', logits)

  def __build_chn_select_ops(self):
    """Build channel selection operations for convolutional layers.

    Returns:
    * chn_select_ops: list of channel selection operations (one per convolutional layer)
    """

    # build layer-wise regression losses
    pattern = re.compile(r'/Conv2D$')
    conv_ops_full = get_ops_by_scope_n_pattern(self.model_scope_full, pattern)
    conv_ops_prnd = get_ops_by_scope_n_pattern(self.model_scope_prnd, pattern)
    reg_losses = []
    for conv_op_full, conv_op_prnd in zip(conv_ops_full, conv_ops_prnd):
      reg_losses += [tf.nn.l2_loss(conv_op_full.outputs[0] - conv_op_prnd.outputs[0])]

    # build layer-wise sampling operations
    conv_info_list = []
    for idx_layer, (conv_op_full, conv_op_prnd) in enumerate(zip(conv_ops_full, conv_ops_prnd)):
      conv_krnl_shape = self.vars_prnd['conv_krnl'][idx_layer].shape
      conv_krnl_prnd_ph = tf.placeholder(
        tf.float32, shape=conv_krnl_shape, name='conv_krnl_prnd_ph_%d' % idx_layer)
      conv_info_list += [{
        'conv_krnl_full': self.vars_full['conv_krnl'][idx_layer],
        'conv_krnl_prnd': self.vars_prnd['conv_krnl'][idx_layer],
        'conv_krnl_prnd_ph': conv_krnl_prnd_ph,
        'update_op': self.vars_prnd['conv_krnl'][idx_layer].assign(conv_krnl_prnd_ph),
        'input_full': conv_op_full.inputs[0],
        'input_prnd': conv_op_prnd.inputs[0],
        'output_full': conv_op_full.outputs[0],
        'output_prnd': conv_op_prnd.outputs[0],
        'strides': conv_op_full.get_attr('strides'),
        'padding': conv_op_full.get_attr('padding').decode('utf-8'),
      }]

    # build meta LASSO/least-square optimization problems
    self.meta_lasso = self.__build_meta_lasso()
    self.meta_lstsq = self.__build_meta_lstsq()

    self.reg_losses = reg_losses
    self.conv_info_list = conv_info_list
    self.nb_conv_layers = len(self.reg_losses)

  def __build_meta_lasso(self):
    """Build a meta LASSO optimization problem."""

    # build a meta LASSO optimization problem
    with tf.variable_scope('meta_lasso'):
      # create placeholders to customize the LASSO problem
      xt_x_ph = tf.placeholder(tf.float32, name='xt_x_ph')
      xt_y_ph = tf.placeholder(tf.float32, name='xt_y_ph')
      mask_ph = tf.placeholder(tf.float32, name='mask_ph')
      gamma = tf.placeholder(tf.float32, shape=[], name='gamma')

      # create variables
      xt_x = tf.get_variable('xt_x', initializer=xt_x_ph, trainable=False, validate_shape=False)
      xt_y = tf.get_variable('xt_y', initializer=xt_y_ph, trainable=False, validate_shape=False)
      mask = tf.get_variable('mask', initializer=mask_ph, trainable=True, validate_shape=False)

      # TF operations
      def prox_mapping(x, thres):
        return tf.where(x > thres, x - thres, tf.where(x < -thres, x + thres, tf.zeros_like(x)))
      mask_gd = mask - FLAGS.cpr_ista_lrn_rate * (tf.matmul(xt_x, mask) - xt_y)
      train_op = mask.assign(prox_mapping(mask_gd, gamma * FLAGS.cpr_ista_lrn_rate))
      init_op = tf.variables_initializer([xt_x, xt_y, mask])

    # pack placeholders, variables, and TF operations into dict
    meta_lasso = {
      'xt_x_ph': xt_x_ph,
      'xt_y_ph': xt_y_ph,
      'mask_ph': mask_ph,
      'gamma': gamma,
      'xt_x': xt_x,
      'xy_y': xt_y,
      'mask': mask,
      'init_op': init_op,
      'train_op': train_op,
    }

    return meta_lasso

  def __build_meta_lstsq(self):
    """Build a meta least-square optimization problem."""

    # build a meta least-square optimization problem
    with tf.variable_scope('meta_lstsq'):
      # create placeholders to customize the least-square problem
      feat_mat_ph = tf.placeholder(tf.float32, name='feat_mat_ph')
      rspn_mat_ph = tf.placeholder(tf.float32, name='rspn_mat_ph')

      # compute the closed-form solution
      wei_mat = tf.linalg.lstsq(feat_mat_ph, rspn_mat_ph, FLAGS.loss_w_dcy)

    # pack placeholders and variables into dict
    meta_lstsq = {
      'feat_mat_ph': feat_mat_ph,
      'rspn_mat_ph': rspn_mat_ph,
      'wei_mat': wei_mat,
    }

    return meta_lstsq

  def __calc_grads_pruned(self, grads_origin):
    """Calculate the mask-pruned gradients.

    Args:
    * grads_origin: list of original gradients

    Returns:
    * grads_pruned: list of mask-pruned gradients
    """

    grads_pruned = []
    conv_krnl_names = [var.name for var in self.vars_prnd['conv_krnl']]
    for grad in grads_origin:
      if grad[1].name not in conv_krnl_names:
        grads_pruned += [grad]
      else:
        idx_mask = conv_krnl_names.index(grad[1].name)
        grads_pruned += [(grad[0] * self.masks[idx_mask], grad[1])]

    return grads_pruned

  def __choose_channels(self):  # pylint: disable=too-many-locals
    """Choose channels for all convolutional layers."""

    # obtain each layer's pruning ratio
    prune_ratios = [FLAGS.cpr_prune_ratio] * self.nb_conv_layers
    if FLAGS.cpr_skip_frst_layer:
      prune_ratios[0] = 0.0
    if FLAGS.cpr_skip_last_layer:
      prune_ratios[-1] = 0.0

    # select channels for all the convolutional layers
    nb_workers = mgw.size() if FLAGS.enbl_multi_gpu else 1
    for idx_layer, (prune_ratio, conv_info) in enumerate(zip(prune_ratios, self.conv_info_list)):
      # skip if no pruning is required
      if prune_ratio == 0.0:
        continue
      if self.is_primary_worker('global'):
        tf.logging.info('layer #%d: pr = %.2f (target)' % (idx_layer, prune_ratio))
        tf.logging.info('kernel shape = {}'.format(conv_info['conv_krnl_prnd'].shape))

      # extract the current layer's information
      conv_krnl_full = self.sess_train.run(conv_info['conv_krnl_full'])
      conv_krnl_prnd = self.sess_train.run(conv_info['conv_krnl_prnd'])
      conv_krnl_prnd_ph = conv_info['conv_krnl_prnd_ph']
      update_op = conv_info['update_op']
      input_full_tf = conv_info['input_full']
      input_prnd_tf = conv_info['input_prnd']
      output_full_tf = conv_info['output_full']
      output_prnd_tf = conv_info['output_prnd']
      strides = conv_info['strides']
      padding = conv_info['padding']
      nb_chns_input = conv_krnl_prnd.shape[2]

      # sample inputs & outputs through multiple mini-batches
      nb_iters_smpl = int(math.ceil(float(FLAGS.cpr_nb_smpl_insts) / FLAGS.batch_size))
      inputs_list = [[] for __ in range(nb_chns_input)]
      outputs_list = []
      for idx_iter in range(nb_iters_smpl):
        inputs_full, inputs_prnd, outputs_full, outputs_prnd = \
          self.sess_train.run([input_full_tf, input_prnd_tf, output_full_tf, output_prnd_tf])
        inputs_smpl, outputs_smpl = self.__smpl_inputs_n_outputs(
          conv_krnl_full, conv_krnl_prnd, inputs_full, inputs_prnd, outputs_full, outputs_prnd, strides, padding)
        for idx_chn_input in range(nb_chns_input):
          inputs_list[idx_chn_input] += [inputs_smpl[idx_chn_input]]
        outputs_list += [outputs_smpl]
      inputs_np_list = [np.vstack(x) for x in inputs_list]
      outputs_np = np.vstack(outputs_list)

      # choose channels via solving the sparsity-constrained regression problem
      conv_krnl_prnd = self.__solve_sparse_regression(
        inputs_np_list, outputs_np, conv_krnl_prnd, prune_ratio)
      self.sess_train.run(update_op, feed_dict={conv_krnl_prnd_ph: conv_krnl_prnd})

      # evaluate the channel pruned model
      if FLAGS.cpr_eval_per_layer:
        if self.is_primary_worker('global'):
          self.__save_model(is_train=True)
          self.evaluate()
        self.auto_barrier()

    # evaluate the final channel pruned model
    if not FLAGS.cpr_eval_per_layer:
      if self.is_primary_worker('global'):
        self.__save_model(is_train=True)
        self.evaluate()
      self.auto_barrier()

  def __smpl_inputs_n_outputs(self, conv_krnl_full, conv_krnl_prnd, inputs_full, inputs_prnd, outputs_full, outputs_prnd, strides, padding):
    """Sample inputs & outputs of sub-regions from full feature maps.

    Args:

    Returns:
    """

    # obtain parameters
    bs = inputs_full.shape[0]
    kh, kw = conv_krnl_full.shape[0], conv_krnl_full.shape[1]
    ih, iw, ic = inputs_full.shape[1], inputs_full.shape[2], inputs_full.shape[3]
    oh, ow, oc = outputs_full.shape[1], outputs_full.shape[2], outputs_full.shape[3]
    if padding == 'VALID':
      ph, pw = 0, 0
    else:
      ph = int(math.ceil((kh - 1) / 2))
      pw = int(math.ceil((kw - 1) / 2))

    # perform zero-padding on input feature maps
    if ph == 0 and pw == 0:
      inputs_full_pad = inputs_full
      inputs_prnd_pad = inputs_prnd
    else:
      inputs_full_pad = np.pad(inputs_full, ((0,), (ph,), (pw,), (0,)), 'constant')
      inputs_prnd_pad = np.pad(inputs_prnd, ((0,), (ph,), (pw,), (0,)), 'constant')

    # sample inputs & outputs of sub-regions
    inputs_smpl_full_list = []
    inputs_smpl_prnd_list = []
    outputs_smpl_full_list = []
    outputs_smpl_prnd_list = []
    for idx_iter in range(FLAGS.cpr_nb_smpl_crops):
      idx_oh = np.random.randint(oh)
      idx_ow = np.random.randint(ow)
      idx_ih_low = idx_oh * strides[1]
      idx_ih_hgh = idx_ih_low + kh
      idx_iw_low = idx_ow * strides[2]
      idx_iw_hgh = idx_iw_low + kw
      inputs_smpl_full_list += [inputs_full_pad[:, idx_ih_low:idx_ih_hgh, idx_iw_low:idx_iw_hgh, :]]
      inputs_smpl_prnd_list += [inputs_prnd_pad[:, idx_ih_low:idx_ih_hgh, idx_iw_low:idx_iw_hgh, :]]
      outputs_smpl_full_list += [np.reshape(outputs_full[:, idx_oh, idx_ow, :], [bs, -1])]
      outputs_smpl_prnd_list += [np.reshape(outputs_prnd[:, idx_oh, idx_ow, :], [bs, -1])]

    # concatenate samples into a single np.array
    inputs_smpl_full = np.concatenate(inputs_smpl_full_list, axis=0)
    inputs_smpl_prnd = np.concatenate(inputs_smpl_prnd_list, axis=0)
    outputs_smpl_full = np.vstack(outputs_smpl_full_list)
    outputs_smpl_prnd = np.vstack(outputs_smpl_prnd_list)

    # validate inputs & outputs
    wei_mat_full = np.reshape(conv_krnl_full, [-1, oc])
    wei_mat_prnd = np.reshape(conv_krnl_prnd, [-1, oc])
    preds_smpl_full = np.matmul(np.reshape(inputs_smpl_full, [-1, kh * kw * ic]), wei_mat_full)
    preds_smpl_prnd = np.matmul(np.reshape(inputs_smpl_prnd, [-1, kh * kw * ic]), wei_mat_prnd)
    err_full = norm(outputs_smpl_full - preds_smpl_full) ** 2 / outputs_smpl_full.shape[0]
    err_prnd = norm(outputs_smpl_prnd - preds_smpl_prnd) ** 2 / outputs_smpl_prnd.shape[0]
    assert err_full < 1e-10, 'unable to recover output feature maps - full (%e)' % err_full
    assert err_prnd < 1e-10, 'unable to recover output feature maps - prnd (%e)' % err_prnd

    # concatenate sampled inputs & outputs arrays
    inputs_smpl = np.split(inputs_smpl_prnd, ic, axis=3)  # one per input channel
    for idx in range(ic):
      inputs_smpl[idx] = np.reshape(inputs_smpl[idx], [-1, kh * kw])
    outputs_smpl = outputs_smpl_full

    return inputs_smpl, outputs_smpl

  def __solve_sparse_regression(self, inputs_np_list, outputs_np, conv_krnl, prune_ratio):
    """Solve the sparsity-constrained regression problem.

    Args:
    * inputs_np_list: list of input feature maps (one per input channel, N x k^2)
    * outputs_np: output feature maps (N x c_o)
    * conv_krnl: initial convolutional kernel (k * k * c_i * c_o)
    * prune_ratio: pruning ratio

    Returns:
    * conv_krnl: updated convolutional kernel (k * k * c_i * c_o)
    """

    # obtain parameters
    bs = outputs_np.shape[0]
    kh, kw, ic, oc = conv_krnl.shape[0], conv_krnl.shape[1], conv_krnl.shape[2], conv_krnl.shape[3]

    # compute the feature matrix & response vector
    rspn_vec_np = np.reshape(outputs_np, [-1, 1])  # N' x 1 (N' = N * c_o)
    feat_mat_np = np.zeros((rspn_vec_np.shape[0], ic))  # N' x c_i
    for idx in range(ic):
      wei_mat = np.reshape(conv_krnl[:, :, idx, :], [kh * kw, oc])
      feat_mat_np[:, idx] = np.matmul(inputs_np_list[idx], wei_mat).ravel()

    # compute <X^T * X> & <X^T * y> in advance
    xt_x_np = np.matmul(feat_mat_np.T, feat_mat_np) / bs
    xt_y_np = np.matmul(feat_mat_np.T, rspn_vec_np) / bs
    mask_np_init = np.ones((ic, 1))

    # solve the LASSO problem
    def __solve_lasso(x):
      self.sess_train.run(self.meta_lasso['init_op'], feed_dict={
        self.meta_lasso['xt_x_ph']: xt_x_np,
        self.meta_lasso['xt_y_ph']: xt_y_np,
        self.meta_lasso['mask_ph']: mask_np_init,
      })
      for __ in range(FLAGS.cpr_ista_nb_iters):
        self.sess_train.run(self.meta_lasso['train_op'], feed_dict={self.meta_lasso['gamma']: x})
      mask_np = self.sess_train.run(self.meta_lasso['mask'])
      nb_chns_nnz = np.count_nonzero(mask_np)
      tf.logging.info('x = %e -> nb_chns_nnz = %d' % (x, nb_chns_nnz))
      return mask_np, nb_chns_nnz

    # determine <gamma> via binary search
    val = 0.1
    nb_chns_nnz_target = int(ic * (1.0 - prune_ratio))
    while True:
      mask_np, nb_chns_nnz = __solve_lasso(val)
      if nb_chns_nnz > nb_chns_nnz_target:
        val *= 2.0
      else:
        break
    lbnd = val / 2.0
    ubnd = val
    while True:
      val = (lbnd + ubnd) / 2.0
      mask_np, nb_chns_nnz = __solve_lasso(val)
      if nb_chns_nnz < nb_chns_nnz_target:
        ubnd = val
      elif nb_chns_nnz > nb_chns_nnz_target:
        lbnd = val
      else:
        break
    tf.logging.info('gamma-final: %e' % val)

    # construct a least-square regression problem
    rspn_mat_np = outputs_np
    bnry_vec_np = (mask_np > 0.0)
    inputs_np_list_msk = [bnry_vec_np[idx] * inputs_np_list[idx] for idx in range(ic)]
    feat_mat_np = np.reshape(
      np.concatenate([np.expand_dims(x, axis=-1) for x in inputs_np_list_msk], axis=-1), [bs, -1])
    wei_mat_np = self.sess_train.run(self.meta_lstsq['wei_mat'], feed_dict={
      self.meta_lstsq['feat_mat_ph']: feat_mat_np,
      self.meta_lstsq['rspn_mat_ph']: rspn_mat_np,
    })
    conv_krnl = np.reshape(wei_mat_np, conv_krnl.shape) * np.reshape(bnry_vec_np, [1, 1, -1, 1])

    return conv_krnl

  def __save_model(self, is_train):
    """Save the current model for training or evaluation.

    Args:
    * is_train: whether to save a model for training
    """

    if is_train:
      save_path = self.saver_prnd_train.save(self.sess_train, FLAGS.cpr_save_path, self.global_step)
    else:
      save_path = self.saver_prnd_eval.save(self.sess_eval, FLAGS.cpr_save_path_eval)
    tf.logging.info('model saved to ' + save_path)

  def __restore_model(self, is_train):
    """Restore a model from the latest checkpoint files.

    Args:
    * is_train: whether to restore a model for training
    """

    save_path = tf.train.latest_checkpoint(os.path.dirname(FLAGS.cpr_save_path))
    if is_train:
      self.saver_prnd_train.restore(self.sess_train, save_path)
    else:
      self.saver_prnd_eval.restore(self.sess_eval, save_path)
    tf.logging.info('model restored from ' + save_path)

  def __monitor_progress(self, summary, log_rslt, idx_iter, time_step):
    """Monitor the training progress.

    Args:
    * summary: summary protocol buffer
    * log_rslt: logging operations' results
    * idx_iter: index of the training iteration
    * time_step: time step between two summary operations
    """

    # write summaries for TensorBoard visualization
    self.sm_writer.add_summary(summary, idx_iter)

    # compute the training speed
    speed = FLAGS.batch_size * FLAGS.summ_step / time_step
    if FLAGS.enbl_multi_gpu:
      speed *= mgw.size()

    # display monitored statistics
    log_str = ' | '.join(['%s = %.4e' % (name, value)
                          for name, value in zip(self.log_op_names, log_rslt)])
    tf.logging.info('iter #%d: %s | speed = %.2f pics / sec' % (idx_iter + 1, log_str, speed))

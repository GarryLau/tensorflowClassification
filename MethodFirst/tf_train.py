"""Build and train model with options for quantization."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

# Can be any nets you want to train
from nets import mobilenet_v1

import tf_input

slim = tf.contrib.slim
flags = tf.app.flags

flags.DEFINE_string('master', '', 'Session master')
flags.DEFINE_integer('task', 0, 'Task')
flags.DEFINE_integer('ps_tasks', 0, 'Number of ps')
flags.DEFINE_integer('batch_size', 32, 'Batch size')
flags.DEFINE_integer('num_classes', 5, 'Number of classes to distinguish')
flags.DEFINE_integer('number_of_steps', 10000,
                     'Number of training steps to perform before stopping')
flags.DEFINE_integer('image_size', 224, 'Input image resolution')
flags.DEFINE_float('depth_multiplier', 1.0, 'Depth multiplier for mobilenet')
flags.DEFINE_bool('quantize', False, 'Quantize training')
flags.DEFINE_string('fine_tune_checkpoint', '/home/lg/projects/mobilenet_v1_1.0_224',
                    'Checkpoint from which to start finetuning.')
flags.DEFINE_string('checkpoint_dir', r'/home/lg/Desktop/finetune/checkpoint',
                    'Directory for writing training checkpoints and logs')
flags.DEFINE_integer('log_every_n_steps', 100, 'Number of steps per log')
flags.DEFINE_integer('save_summaries_secs', 100,
                     'How often to save summaries, secs')
flags.DEFINE_integer('save_interval_secs', 100,
                     'How often to save checkpoints, secs')
flags.DEFINE_string('checkpoint_exclude_scopes', 'MobilenetV1/Logits/Conv2d_1c_1x1/weights,MobilenetV1/Logits/Conv2d_1c_1x1/biases',
                    'Comma-separated list of scopes of variables to exclude when restoring '
                    'from a checkpoint.')

FLAGS = flags.FLAGS

_LEARNING_RATE_DECAY_FACTOR = 0.94


def get_learning_rate():
  if FLAGS.fine_tune_checkpoint:
    # If we are fine tuning a checkpoint we need to start at a lower learning
    # rate since we are farther along on training.
    return 1e-4
  else:
    return 0.045


def get_quant_delay():
  if FLAGS.fine_tune_checkpoint:
    # We can start quantizing immediately if we are finetuning.
    return 0
  else:
    # We need to wait for the model to train a bit before we quantize if we are
    # training from scratch.
    return 250000


def build_model():
  """Builds graph for model to train with rewrites for quantization.

  Returns:
    g: Graph with fake quantization ops and batch norm folding suitable for
    training quantized weights.
    train_tensor: Train op for execution during training.
  """
  g = tf.Graph()
  with g.as_default(), tf.device(
      tf.train.replica_device_setter(FLAGS.ps_tasks)):

    #Reads in data and/or performs pre-processing on the images.
    inputs, labels, _ = tf_input.input(is_random=True, is_training=True)
    labels = labels - 1
    labels = slim.one_hot_encoding(labels, FLAGS.num_classes)

    with slim.arg_scope(mobilenet_v1.mobilenet_v1_arg_scope(is_training=True)):
      logits, _ = mobilenet_v1.mobilenet_v1(
          inputs,
          is_training=True,
          depth_multiplier=FLAGS.depth_multiplier,
          num_classes=FLAGS.num_classes)

    tf.losses.softmax_cross_entropy(labels, logits)

    # Call rewriter to produce graph with fake quant ops and folded batch norms
    # quant_delay delays start of quantization till quant_delay steps, allowing
    # for better model accuracy.
    if FLAGS.quantize:
      tf.contrib.quantize.create_training_graph(quant_delay=get_quant_delay())

    total_loss = tf.losses.get_total_loss(name='total_loss')
    # Configure the learning rate using an exponential decay.
    num_epochs_per_decay = 2.5
    data_size = tf_input.TRAINING_SET_SIZE
    decay_steps = int(data_size / FLAGS.batch_size * num_epochs_per_decay)

    learning_rate = tf.train.exponential_decay(
        get_learning_rate(),
        tf.train.get_or_create_global_step(),
        decay_steps,
        _LEARNING_RATE_DECAY_FACTOR,
        staircase=True)
    opt = tf.train.GradientDescentOptimizer(learning_rate)

    train_tensor = slim.learning.create_train_op(
        total_loss,
        optimizer=opt)

  slim.summaries.add_scalar_summary(total_loss, 'total_loss', 'losses')
  slim.summaries.add_scalar_summary(learning_rate, 'learning_rate', 'training')
  return g, train_tensor


def get_checkpoint_init_fn():
  """Returns the checkpoint init_fn if the checkpoint is provided."""
  if FLAGS.fine_tune_checkpoint:
    exclusions = []
    if FLAGS.checkpoint_exclude_scopes:
        exclusions = [scope.strip()
                      for scope in FLAGS.checkpoint_exclude_scopes.split(',')]
    variables_to_restore = []
    for var in slim.get_model_variables():
        excluded = False
        for exclusion in exclusions:
            if var.op.name.startswith(exclusion):
                excluded = True
                break
        if not excluded:
            variables_to_restore.append(var)
    global_step_reset = tf.assign(tf.train.get_or_create_global_step(), 0)
    # When restoring from a floating point model, the min/max values for
    # quantized weights and activations are not present.
    # We instruct slim to ignore variables that are missing during restoration
    # by setting ignore_missing_vars=True

    if tf.gfile.IsDirectory(FLAGS.fine_tune_checkpoint):
        fine_checkpoint_path = tf.train.latest_checkpoint(FLAGS.fine_tune_checkpoint)
    else:
        fine_checkpoint_path = FLAGS.fine_tune_checkpoint
    slim_init_fn = slim.assign_from_checkpoint_fn(
        fine_checkpoint_path,
        variables_to_restore,
        ignore_missing_vars=True)

    def init_fn(sess):
      slim_init_fn(sess)
      # If we are restoring from a floating point model, we need to initialize
      # the global step to zero for the exponential decay to result in
      # reasonable learning rates.
      sess.run(global_step_reset)
    return init_fn
  else:
    return None


def train_model():
  """Trains model."""
  g, train_tensor = build_model()
  with g.as_default():
    slim.learning.train(
        train_tensor,
        FLAGS.checkpoint_dir,
        is_chief=(FLAGS.task == 0),
        master=FLAGS.master,
        log_every_n_steps=FLAGS.log_every_n_steps,
        graph=g,
        number_of_steps=FLAGS.number_of_steps,
        save_summaries_secs=FLAGS.save_summaries_secs,
        save_interval_secs=FLAGS.save_interval_secs,
        init_fn=get_checkpoint_init_fn(),
        global_step=tf.train.get_global_step())


def main(unused_arg):
  train_model()


if __name__ == '__main__':
  tf.app.run(main)

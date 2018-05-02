"""Validate model with options for quantization."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math
import tensorflow as tf

# Can be any nets you want to eval
from nets import mobilenet_v1

import tf_input

slim = tf.contrib.slim
flags = tf.app.flags

flags.DEFINE_string('master', '', 'Session master')
flags.DEFINE_integer('batch_size', 32, 'Batch size')
flags.DEFINE_integer('num_classes', 5, 'Number of classes to distinguish')
flags.DEFINE_integer('num_examples', 350, 'Number of examples to evaluate')
flags.DEFINE_float('depth_multiplier', 1.0, 'Depth multiplier for mobilenet')
flags.DEFINE_bool('quantize', False, 'Quantize training')
flags.DEFINE_string('checkpoint_dir',  r'/home/lg/Desktop/finetune/checkpoint/model.ckpt-10000',
                    'The directory for checkpoints')
flags.DEFINE_string('eval_dir',  r'/home/lg/Desktop/finetune/eval',
                    'Directory for writing eval event logs')

FLAGS = flags.FLAGS


def metrics(logits, labels):
  """Specify the metrics for eval.

  Args:
    logits: Logits output from the graph.
    labels: Ground truth labels for inputs.

  Returns:
     Eval Op for the graph.
  """
  labels = tf.squeeze(labels)
  names_to_values, names_to_updates = slim.metrics.aggregate_metric_map({
      'Accuracy': tf.metrics.accuracy(tf.argmax(logits, 1), labels),
      'Recall_5': tf.metrics.recall_at_k(labels, logits, 1),
  })
  for name, value in names_to_values.items():
    slim.summaries.add_scalar_summary(
        value, name, prefix='eval', print_summary=True)
  return list(names_to_updates.values())


def build_model():
  """Build the model for evaluation.

  Returns:
    g: graph with rewrites after insertion of quantization ops and batch norm
    folding.
    eval_ops: eval ops for inference.
    variables_to_restore: List of variables to restore from checkpoint.
  """
  g = tf.Graph()
  with g.as_default():
    #Reads in data
    inputs, labels, _ = tf_input.input(is_random=True, is_training=True)
    labels = labels - 1

    scope = mobilenet_v1.mobilenet_v1_arg_scope(
        is_training=False, weight_decay=0.0)
    with slim.arg_scope(scope):
      logits, _ = mobilenet_v1.mobilenet_v1(
          inputs,
          is_training=False,
          depth_multiplier=FLAGS.depth_multiplier,
          num_classes=FLAGS.num_classes)
    if FLAGS.quantize:
      tf.contrib.quantize.create_eval_graph()

    eval_ops = metrics(logits, labels)

  return g, eval_ops


def eval_model():
  """Evaluates model."""
  g, eval_ops = build_model()
  with g.as_default():
    num_batches = math.ceil(FLAGS.num_examples / float(FLAGS.batch_size))
    slim.evaluation.evaluate_once(
        FLAGS.master,
        FLAGS.checkpoint_dir,
        logdir=FLAGS.eval_dir,
        num_evals=num_batches,
        eval_op=eval_ops)


def main(unused_arg):
    eval_model()


if __name__ == '__main__':
  tf.app.run(main)

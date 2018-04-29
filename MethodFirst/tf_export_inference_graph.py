"""Saves out a GraphDef containing the architecture of the model."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math
import tensorflow as tf

# Can be any nets you want to export
from nets import mobilenet_v1

slim = tf.contrib.slim
flags = tf.app.flags

flags.DEFINE_integer('num_classes', 5, 'Number of classes to distinguish')
flags.DEFINE_float('depth_multiplier', 1.0, 'Depth multiplier for mobilenet')
flags.DEFINE_bool('quantize', False, 'Quantize training')
flags.DEFINE_string('eval_graph_file',  r'/home/lg/Desktop/finetune/mobilenet_v1_eval.pbtxt',
                    'Directory for writing eval graph.')

FLAGS = flags.FLAGS

IMAGE_SIZE = 224
CHANNEL = 3

def export_eval_pbtxt():
  """Export eval.pbtxt."""
  g = tf.Graph()
  with g.as_default():
    inputs = tf.placeholder(dtype=tf.float32,shape=[None, IMAGE_SIZE, IMAGE_SIZE, CHANNEL])
    scope = mobilenet_v1.mobilenet_v1_arg_scope(
        is_training=False, weight_decay=0.0)
    with slim.arg_scope(scope):
      _, _ = mobilenet_v1.mobilenet_v1(
          inputs,
          is_training=False,
          depth_multiplier=FLAGS.depth_multiplier,
          num_classes=FLAGS.num_classes)
    if FLAGS.quantize:
      tf.contrib.quantize.create_eval_graph()
    with tf.Session() as sess:
          with open(FLAGS.eval_graph_file, 'w') as f:
            f.write(str(g.as_graph_def()))

def main(unused_arg):
    export_eval_pbtxt()


if __name__ == '__main__':
  tf.app.run(main)
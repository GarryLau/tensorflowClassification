"""Predict one image."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import os.path
import sys

import numpy as np
import tensorflow as tf

FLAGS = None
IMAGE_SIZE = 224
CHANNEL = 3


def create_graph():
  """Creates a graph from saved GraphDef file and returns a saver."""
  # Creates graph from saved graph_def.pb.
  with tf.gfile.FastGFile(os.path.join(FLAGS.model_dir, r'/home/lg/Desktop/finetune/frozen_mobilenet_v1_224.pb'), 'rb') as f:
    graph_def = tf.GraphDef()
    graph_def.ParseFromString(f.read())
    _ = tf.import_graph_def(graph_def,return_elements=['MobilenetV1/Predictions/Reshape_1:0'], name='lg')

def run_inference_on_image(image):
  """Runs inference on an image.
  Args:
    image: Image file name.
  Returns:
    Nothing
  """
  if not tf.gfile.Exists(image):
    tf.logging.fatal('File does not exist %s', image)

  # Read an image
  image_data = tf.gfile.FastGFile(image, 'rb').read()
  img_data_jpg = tf.image.decode_jpeg(image_data)   # Decode image
  img_data_jpg = tf.image.convert_image_dtype(img_data_jpg, dtype=tf.float32) # Convert uint8 to float32
  img_data_jpg = tf.image.resize_image_with_crop_or_pad(img_data_jpg,IMAGE_SIZE,IMAGE_SIZE)

  # Creates graph from saved GraphDef.
  create_graph()

  with tf.Session() as sess:
    image_data = img_data_jpg.eval().reshape(-1,IMAGE_SIZE,IMAGE_SIZE,CHANNEL)
    softmax_tensor = sess.graph.get_tensor_by_name('lg/MobilenetV1/Predictions/Reshape_1:0')
    predictions = sess.run(softmax_tensor, {'lg/Placeholder:0': image_data})
    predictions = np.squeeze(predictions)
    print('predictions: ',predictions)
    # Read the labels from label.txt.
    label_path = os.path.join(FLAGS.model_dir, '/home/lg/projects/labels.txt')
    label = np.loadtxt(fname=label_path,dtype=str)

    top_k = predictions.argsort()[-FLAGS.num_top_predictions:][::-1]
    for node_id in top_k:
      label_string = label[node_id]
      score = predictions[node_id]
      print('%s (score = %.5f)' % (label_string, score))


def main(_):
  image = FLAGS.image_file
  run_inference_on_image(image)


if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  # graph_def.pb: Binary representation of the GraphDef protocol buffer.
  # label.txt: the labels according to data tfrecord
  parser.add_argument(
      '--model_dir',
      type=str,
      default='',
      help='Path to graph_def.pb and label.txt'
  )
  parser.add_argument(
      '--image_file',
      type=str,
      default=r'/home/lg/projects/data/flower_photos/daisy/15207766_fc2f1d692c_n.jpg',
      help='Absolute path to image file.'
  )
  parser.add_argument(
      '--num_top_predictions',
      type=int,
      default=2,
      help='Display this many predictions.'
  )
  FLAGS, unparsed = parser.parse_known_args()
tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)
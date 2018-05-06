"""Predict many images, and output the results to txt."""

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
  with tf.gfile.FastGFile(os.path.join(FLAGS.model_dir,
                                       r'/home/lg/Desktop/mobilenet_v1_100_224/frozen_mobilenet_v1_100_224_quantized.pb'), 'rb') as f:
    graph_def = tf.GraphDef()
    graph_def.ParseFromString(f.read())
    _ = tf.import_graph_def(graph_def,return_elements=['MobilenetV1/Predictions/Reshape_1:0'], name='lg')

def run_inference_on_image(image_txt):
  """Runs inference on a list of images.
  Args:
    image_txt: The path of images list.
  Returns:
    Nothing
  """

  # Creates graph from saved GraphDef.
  create_graph()

  if not tf.gfile.Exists(image_txt):
    tf.logging.fatal('File does not exist %s', image_txt)

  # create the txt file to save predict results.
  f = open(r'/home/lg/Desktop/frozen_mobilenet_v1_100_224_quantized_prediction.txt','w')
  image_list = np.loadtxt(image_txt,dtype=str,delimiter=' ')
  # Read the labels from label.txt.
  label_path = os.path.join(FLAGS.model_dir, '/home/lg/Desktop/label.txt')
  label = np.loadtxt(fname=label_path, dtype=str)

  image_placeholder = tf.placeholder(dtype=tf.string)
  img_data_jpg = tf.image.decode_jpeg(image_placeholder)  # Decode image
  img_data_jpg = tf.image.convert_image_dtype(img_data_jpg, dtype=tf.float32)  # Change the image's dtype
  img_data_jpg = tf.image.resize_image_with_crop_or_pad(img_data_jpg, IMAGE_SIZE, IMAGE_SIZE)

  with tf.Session() as sess:
      for i in range(len(image_list)):
        image = image_list[i][0]
        image_data = tf.gfile.FastGFile(image, 'rb').read()
        img_data_feed = sess.run(img_data_jpg,{image_placeholder:image_data})
        img_data_feed = img_data_feed.reshape(-1,IMAGE_SIZE,IMAGE_SIZE,CHANNEL)

        softmax_tensor = sess.graph.get_tensor_by_name('lg/MobilenetV1/Predictions/Reshape_1:0')
        predictions = sess.run(softmax_tensor, {'lg/Placeholder:0': img_data_feed})
        predictions = np.squeeze(predictions)
        #print('predictions: ',predictions)
        top_k = predictions.argsort()[-FLAGS.num_top_predictions:][::-1]

        for node_id in top_k:
          label_string = label[node_id]
          score = predictions[node_id]
          #print('%s (score = %.5f)' % (label_string, score))
        f.write(image + ' ' + label_string + ' ' + str(score) + '\n')
  f.close()
  print('Done!')


def main(_):
  image_txt = FLAGS.image_txt
  run_inference_on_image(image_txt)


if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  # graph_def.pb: Binary representation of the GraphDef protocol buffer.
  # label.txt: the labels according to data tfrecord
  parser.add_argument(
      '--model_dir',
      type=str,
      default='/tmp/imagenet',
      help='Path to graph_def.pb and label.txt'
  )
  parser.add_argument(
      '--image_txt',
      type=str,
      default='/home/lg/Desktop/prediction_image_txt.txt',
      help='Absolute path to image txt file.'
  )
  parser.add_argument(
      '--num_top_predictions',
      type=int,
      default=1,
      help='Display this many predictions.'
  )
  FLAGS, unparsed = parser.parse_known_args()
  tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)

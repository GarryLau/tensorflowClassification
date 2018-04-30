from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from datetime import datetime
import time
import tensorflow as tf
import tf_inference
import tf_input
from nets import inception_v3

slim = tf.contrib.slim

FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_string('train_dir', r'/home/lg/Desktop/method/checkpoint',
                           """Directory where to write event logs """
                           """and checkpoint.""")
tf.app.flags.DEFINE_integer('max_steps', 10000,
                            """Number of batches to run.""")
tf.app.flags.DEFINE_boolean('log_device_placement', False,
                            """Whether to log device placement.""")
tf.app.flags.DEFINE_integer('log_frequency', 100,
                            """How often to log results to the console.""")

NUM_CLASSES = 5


def train():
    """Train classification for a number of steps."""
    with tf.Graph().as_default():
        # Returns and create (if necessary) the global step tensor.
        global_step = tf.train.get_or_create_global_step()
        # Get images and labels for datasets.
        # Force input pipeline to CPU:0 to avoid operations sometimes ending up on GPU and resulting in a slow down.
        with tf.device('/cpu:0'):
            # images:[batch size size channel], labels:[batch], filenames: nouseful
            images, labels, _ = tf_input.input(is_random=True, is_training=True)
            labels = labels - 1

        # Build a Graph that computes the logits predictions from the inference model.
        # Attentation please, train mobilenet directly maybe very very difficult to convergence,
        # you should do retrain(transfer learning)
        with slim.arg_scope(inception_v3.inception_v3_arg_scope()):
            logits, _ = inception_v3.inception_v3(inputs=images, is_training=True, num_classes=NUM_CLASSES)
        # Or for simply
        # logits, _ = inception_v3.inception_v3(inputs=images, num_classes=NUM_CLASSES, is_training=True)

        # Calculate loss.
        loss = tf_inference.loss(logits, labels)
        # Build a Graph that trains the model with one batch of examples and
        # updates the model parameters.
        train_op = tf_inference.train(loss, global_step)

        class _LoggerHook(tf.train.SessionRunHook):
            """Logs loss and runtime."""

            def begin(self):
                self._step = -1
                self._start_time = time.time()

            def before_run(self, run_context):
                self._step += 1
                return tf.train.SessionRunArgs(loss)  # Asks for loss value.

            def after_run(self, run_context, run_values):
                if self._step % FLAGS.log_frequency == 0:
                    current_time = time.time()
                    duration = current_time - self._start_time
                    self._start_time = current_time

                    loss_value = run_values.results
                    examples_per_sec = FLAGS.log_frequency * FLAGS.batch_size / duration
                    sec_per_batch = float(duration / FLAGS.log_frequency)

                    format_str = ('%s: step %d, loss = %.2f (%.1f examples/sec; %.3f '
                                  'sec/batch)')
                    print(format_str % (datetime.now(), self._step, loss_value,
                                        examples_per_sec, sec_per_batch))


        with tf.train.MonitoredTrainingSession(
                checkpoint_dir=FLAGS.train_dir,
                hooks=[tf.train.StopAtStepHook(last_step=FLAGS.max_steps),
                       tf.train.NanTensorHook(loss),
                       _LoggerHook()],
                config=tf.ConfigProto(log_device_placement=FLAGS.log_device_placement)) as mon_sess:
            while not mon_sess.should_stop():
                mon_sess.run(train_op)


def main(argv=None):
    train()


if __name__ == '__main__':
    tf.app.run()
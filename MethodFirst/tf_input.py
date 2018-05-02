import os.path
import tensorflow as tf

# Basic model parameters as external flags.
FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_string('data_dir', r'/home/lg/Desktop/flower_photos',
                           """Path to the train data and eval data directory.""")
tf.app.flags.DEFINE_integer('image_size', 224, 'Input image resolution')
tf.app.flags.DEFINE_boolean('use_fp16', False,
                            """Train the model using fp16.""")

# Constants used for dealing with the files, matches convert_to_records.
TRAIN_FILE = 'train-00000-of-00001.tfrecord'
VALIDATION_FILE = 'validation-00000-of-00001.tfrecord'
TRAINING_SET_SIZE = 2370


# image object from protobuf
class _image_object:
    def __init__(self):
        self.image = tf.Variable([], dtype = tf.string,trainable=False)
        self.height = tf.Variable([], dtype = tf.int64,trainable=False)
        self.width = tf.Variable([], dtype = tf.int64,trainable=False)
        self.filename = tf.Variable([], dtype = tf.string,trainable=False)
        self.label = tf.Variable([], dtype = tf.int32,trainable=False)

def read_and_decode(filename_queue):
    reader = tf.TFRecordReader()
    _, serialized_example = reader.read(filename_queue)
    features = tf.parse_single_example(serialized_example, features = {
        "image/encoded": tf.FixedLenFeature([], tf.string),
        "image/height": tf.FixedLenFeature([], tf.int64),
        "image/width": tf.FixedLenFeature([], tf.int64),
        "image/filename": tf.FixedLenFeature([], tf.string),
        "image/class/label": tf.FixedLenFeature([], tf.int64),})
    image_encoded = features["image/encoded"]
    image_raw = tf.image.decode_jpeg(image_encoded, channels=3)
    image_object = _image_object()
    image_object.image = tf.image.resize_image_with_crop_or_pad(image_raw, FLAGS.image_size, FLAGS.image_size)
    image_object.height = features["image/height"]
    image_object.width = features["image/width"]
    image_object.filename = features["image/filename"]
    image_object.label = tf.cast(features["image/class/label"], tf.int64)
    return image_object

def input(is_random = True, is_training = True):
    filenames = [os.path.join(FLAGS.data_dir, TRAIN_FILE if is_training else VALIDATION_FILE)]
    for f in filenames:
        if not tf.gfile.Exists(f):
            raise ValueError("Failed to find file: " + f)
    filename_queue = tf.train.string_input_producer(filenames)
    image_object = read_and_decode(filename_queue)
    image = image_object.image
    # Convert uint8 to float, should not use tf.cast()
    image = tf.image.convert_image_dtype(image, dtype= tf.float32)
    # You can do some data augmentation here.
    label = image_object.label
    filename = image_object.filename

    if(is_random):
        min_fraction_of_examples_in_queue = 0.4
        min_queue_examples = int(TRAINING_SET_SIZE * min_fraction_of_examples_in_queue)
        print("Filling queue with %d images before starting to train. " "This will take a few minutes." % min_queue_examples)
        num_preprocess_threads = 1
        image_batch, label_batch, filename_batch = tf.train.shuffle_batch(
            [image, label, filename],
            batch_size = FLAGS.batch_size,
            num_threads = num_preprocess_threads,
            capacity = min_queue_examples + 3 * FLAGS.batch_size,
            min_after_dequeue = min_queue_examples)
        if FLAGS.use_fp16:
            image_batch = tf.cast(image_batch, tf.float16)
            label_batch = tf.cast(label_batch, tf.float16)
        return image_batch, label_batch, filename_batch
    else:
        image_batch, label_batch, filename_batch = tf.train.batch(
            [image, label, filename],
            batch_size = FLAGS.batch_size,
            num_threads = 1)
        if FLAGS.use_fp16:
            image_batch = tf.cast(image_batch, tf.float16)
            label_batch = tf.cast(label_batch, tf.float16)
        return image_batch, label_batch, filename_batch

    
def run_training():
    images_batch, labels_batch, filenames_batch = input()   # use default parameters
def main(_):
  run_training()


if __name__ == '__main__':
  tf.app.run()

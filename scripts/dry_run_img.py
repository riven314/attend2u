# python dry_run_img.py --gpu_id 0 --batch_size 32 --input_fname ../../data/instagram/caption_dataset/train.txt
import os

import numpy as np
from tqdm import tqdm

import tensorflow as tf
from tensorflow.python.framework import ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import gen_image_ops
from tensorflow.python.ops import string_ops
from tensorflow.contrib.slim.python.slim.nets import resnet_v1

import vgg_preprocessing
from utils import read_target_imgs


slim = tf.contrib.slim
FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_string(
    "image_dir",
    os.path.join('..', '..', 'data', 'Instagram', 'images'),
    ""
)
tf.app.flags.DEFINE_string(
    "input_fname",
    "../../data/Instagram/caption_dataset/train.txt",
    ""
)
tf.app.flags.DEFINE_integer("batch_size", 32, "Batch size to use")
tf.app.flags.DEFINE_string("gpu_id", "0", "GPU id to use")
tf.app.flags.DEFINE_string(
    "output_dir",
    '/userhome/34/h3509807/MadeWithML/data/Instagram/resnet_pool5_features',
    "Output directory to save resnet features"
)


def decode_image(contents, channels = None, name = None):
  with ops.name_scope(name, 'decode_image'):
    if channels not in (None, 0, 1, 3):
      raise ValueError('channels must be in (None, 0, 1, 3)')
    substr = string_ops.substr(contents, 0, 4)
    
    def _png():
      return gen_image_ops.decode_png(contents, channels)
    
    def _jpeg():
      return gen_image_ops.decode_jpeg(contents, channels)

    is_png = math_ops.equal(substr, b'\211PNG', name = 'is_png')
    return control_flow_ops.cond(is_png, _png, _jpeg, name = 'cond_jpeg')


if __name__ == '__main__':
  outs = dict()
  os.environ["CUDA_VISIBLE_DEVICES"] = FLAGS.gpu_id

  if not tf.gfile.Exists(FLAGS.output_dir):
    tf.gfile.MakeDirs(FLAGS.output_dir)

  filenames = read_target_imgs(FLAGS.input_fname)
  filenames = [os.path.join(FLAGS.image_dir, fn) for fn in filenames]  
  print(f'total image files: {len(filenames)}')

  with tf.Graph().as_default() as g:
    filename_queue = tf.train.string_input_producer(filenames)
    reader = tf.WholeFileReader()

    # embed print node into computation graph
    key, value = reader.read(filename_queue)
    #print_value = tf.print(value, [value])

    image = decode_image(value, channels = 3)
    #image = tf.image.decode_jpeg(value, channels = 3) # pre-clean your images before run
    image_size = resnet_v1.resnet_v1.default_image_size
    processed_image = vgg_preprocessing.preprocess_image(
        image, image_size, image_size, is_training=False
    )
    processed_images, keys = tf.train.batch(
        [processed_image, key],
        FLAGS.batch_size,
        num_threads=8, capacity=8*FLAGS.batch_size*5,
        allow_smaller_final_batch=True
    )

    with tf.Session() as sess:
      coord = tf.train.Coordinator()
      threads = tf.train.start_queue_runners(coord=coord)
      for step in tqdm(
        range(int(len(filenames) / FLAGS.batch_size) + 1), 
        ncols = 70
      ):
        if coord.should_stop():
          break

        print_val = sess.run(key).decode()
        print(print_val)
        file_names = sess.run([keys])

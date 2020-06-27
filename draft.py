import os

import numpy as np
import tensorflow as tf
import scripts.vgg_preprocessing as vgg
from tensorflow.contrib.slim.python.slim.nets import resnet_v1

BATCH_SIZE = 3
IMG_DIR = os.path.join('cleansing', 'images')
CKPT = os.path.join('scripts', 'resnet_v1_101.ckpt')


with tf.Graph().as_default() as graph:

    filenames = [os.path.join(IMG_DIR, fn) for fn in os.listdir(IMG_DIR)[:50]]
    filename_queue = tf.train.string_input_producer(filenames)
    reader = tf.WholeFileReader()

    key, value = reader.read(filename_queue)
    image = tf.image.decode_jpeg(value, channels = 3)
    processed_image = vgg.preprocess_image(
        image, 224, 224, is_training = False
    )
    processed_images, keys = tf.train.batch(
        [processed_image, key], BATCH_SIZE, 
        num_threads = 4, capacity = 8*4*5,
        # last batch has duplicate on last element
        allow_smaller_final_batch = True 
    )


    with tf.contrib.slim.arg_scope(resnet_v1.resnet_arg_scope()):
        net, end_points = resnet_v1.resnet_v1_101(
            processed_images, num_classes = 1000, is_training = False
        )
        init_fn = tf.contrib.slim.assign_from_checkpoint_fn(
            CKPT, tf.contrib.slim.get_model_variables()
        )
        pool5 = graph.get_operation_by_name('resnet_v1_101/pool5').outputs[0]
        pool5 = tf.transpose(pool5, perm = [0, 3, 1, 2])


        outs = dict()
        with tf.Session() as sess:
            init_fn(sess)
            coord = tf.train.Coordinator()
            threads = tf.train.start_queue_runners(coord = coord)

            for i in range(int(len(filenames)/ BATCH_SIZE) + 1):
                batch, file_names, model_out = sess.run([processed_images, keys, pool5])
                outs[i] = {
                    'images': batch, 'filenames': file_names, 'outputs': model_out
                    }
                i += 1
                if coord.should_stop():
                    break

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import sys
import time
import os
import imtools as its
from sklearn.metrics import confusion_matrix

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

os.environ["CUDA_VISIBLE_DEVICES"]="-1"


def load_graph(model_file):
  graph = tf.Graph()
  graph_def = tf.GraphDef()

  with open(model_file, "rb") as f:
    graph_def.ParseFromString(f.read())
  with graph.as_default():
    tf.import_graph_def(graph_def)

  return graph

def read_tensor_from_image_file(file_name, input_height=299, input_width=299,
				input_mean=0, input_std=255):
  input_name = "file_reader"
  output_name = "normalized"
  file_reader = tf.read_file(file_name, input_name)
  if file_name.endswith(".png"):
    image_reader = tf.image.decode_png(file_reader, channels = 3,
                                       name='png_reader')
  elif file_name.endswith(".gif"):
    image_reader = tf.squeeze(tf.image.decode_gif(file_reader,
                                                  name='gif_reader'))
  elif file_name.endswith(".bmp"):
    image_reader = tf.image.decode_bmp(file_reader, name='bmp_reader')
  else:
    image_reader = tf.image.decode_jpeg(file_reader, channels = 3,
                                        name='jpeg_reader')
  float_caster = tf.cast(image_reader, tf.float32)
  dims_expander = tf.expand_dims(float_caster, 0);
  resized = tf.image.resize_bilinear(dims_expander, [input_height, input_width])
  normalized = tf.divide(tf.subtract(resized, [input_mean]), [input_std])
  sess = tf.Session()
  result = sess.run(normalized)

  return result

def load_labels(label_file):
  label = []
  proto_as_ascii_lines = tf.gfile.GFile(label_file).readlines()
  for l in proto_as_ascii_lines:
    label.append(l.rstrip())
  return label

def get_imlist(path):
    return [os.path.join(path,f) for f in os.listdir(path) if f.endswith('.png')]

if __name__ == "__main__":

  model_file = "tf_files/retrained_graph.pb"
  label_file = "tf_files/retrained_labels.txt"
  graph = load_graph(model_file)
  labels = load_labels(label_file)
  input_height = 128
  input_width = 128
  input_mean = 128
  input_std = 128
  input_layer = "input"
  output_layer = "final_result"
  input_name = "import/" + input_layer
  output_name = "import/" + output_layer
  input_operation = graph.get_operation_by_name(input_name);
  output_operation = graph.get_operation_by_name(output_name);
  y_true = []
  y_pred = []


  for file_index in range(4):

      label_index_str = str(file_index)
      imglist = get_imlist('tf_files/test_photos/'+label_index_str)


      for infile in imglist:
          file_name = os.path.splitext(infile)[0] + ".png"
          # file_name = "tf_files/test_photos/3/da (91).png"
          t = read_tensor_from_image_file(file_name,
                                          input_height=input_height,
                                          input_width=input_width,
                                          input_mean=input_mean,
                                          input_std=input_std)
          with tf.Session(graph=graph) as sess:
              results = sess.run(output_operation.outputs[0],
                                 {input_operation.outputs[0]: t})
          results = np.squeeze(results)
          top_k = results.argsort()[-5:][::-1]

          y_true.append(file_index)
          pred_index = top_k[0]
          y_pred.append(pred_index)
          print(infile)
          template = "{} (score={:0.5f})"
          for i in top_k:
              print(template.format(labels[i], results[i]))


  C = confusion_matrix(y_true, y_pred)
  C_n = C.astype('float') / C.sum(axis=1)[:, np.newaxis]

  plt.show()

  print(C)
  print(C_n)
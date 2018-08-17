import os
import urllib.request
from zipfile import ZipFile
import tensorflow as tf
import numpy as np

# Mean value for images for inception model
INCEPTION_MEAN = 177.0

# URL for the model graph
url = 'https://storage.googleapis.com/download.tensorflow.org/models/inception5h.zip'

# Paths
model_dir = './model'
model_zip_file = os.path.split(url)[-1]
model_zip_path = os.path.join(model_dir, model_zip_file)
model_path = os.path.join(model_dir, 'tensorflow_inception_graph.pb')


def download_model():

    print('Downloading model....')

    # Check for model directory
    if not os.path.exists(model_dir):
        os.mkdir(model_path)

    # Check for the path existance
    if not os.path.exists(model_path):

        # Get the model
        model_data = urllib.request.urlopen(url)

        # Create zipfile
        with open(model_zip_path, 'wb') as zip_f:
            zip_f.write(model_data.read())

        # Extract the zipfile
        with ZipFile(model_zip_path, 'r') as zip_ref:
            zip_ref.extractall(model_dir)

        # Delete the zipfile
        os.remove(model_zip_path)

    print('Model downloaded')


class Inception5H:

    layer_names = ['import/conv2d0', 'import/conv2d1', 'import/conv2d2',
                   'import/mixed3a', 'import/mixed3b',
                   'import/mixed4a', 'import/mixed4b', 'import/mixed4c', 'import/mixed4d', 'import/mixed4e',
                   'import/mixed5a', 'import/mixed5b']

    def __init__(self):

        # Create the graph
        self.graph = tf.Graph()

        with self.graph.as_default():
            # Read the graph
            with tf.gfile.FastGFile(model_path, 'rb') as f:

                # Create graph-def
                graph_def = tf.GraphDef()
                # Write to the graph-def
                graph_def.ParseFromString(f.read())

            # Import the graph-def
            self.input = tf.placeholder(tf.float32, name='input')
            tf.import_graph_def(graph_def, {'input': self.input})

        # List all the tensors
        # _operations = [op.name for op in self.graph.get_operations() if op.name in self.layer_names]
        self.layers = [self.graph.get_tensor_by_name(name + ':0') for name in self.layer_names]
        self.features = [layer.shape[-1] for layer in self.layers]

    def preprocess_image(self, image):
        image = image[np.newaxis, ...]
        return image - INCEPTION_MEAN

    def depreprocess_image(self, image):
        image = np.squeeze(image, axis=0)
        return image + INCEPTION_MEAN

    def get_feed_dict(self, image):
        feed_dict = {self.input: image}
        return feed_dict

    def get_gradient(self, tensor):
        with self.graph.as_default():

            # Transform the tensor
            tensor = tf.reduce_mean(tf.square(tensor))

            # Get the gradientes
            gradient = tf.gradients(tensor, self.input)[0]

            return gradient

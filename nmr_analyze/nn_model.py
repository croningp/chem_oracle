import tensorflow as tf
import math
import numpy as np
from utils import *
from matplotlib import pyplot as plt


def new_weights(shape, name=None):
    stddev = 1 / math.sqrt(shape[-1])
    return tf.Variable(
        tf.truncated_normal(shape, stddev=stddev), dtype=tf.float32, name=name
    )


def new_biases(shape, name=None):
    """Create new biases
    Args:
        shape: an integer for the shape
    Returns:
        A zero initializer biases of shape [shape]
    """
    return tf.Variable(tf.zeros(shape=[shape]), dtype=tf.float32, name=name)


def noisy(data, noise_level):
    """ Change the intensity of spectra"""
    batch_size = tf.shape(data)[0]
    noise = 1.0 + noise_level * tf.random_normal([batch_size, 2, 1, 1], stddev=0.5)
    data = data * noise
    return data


def no_noise(data):
    return data


# A simple function for creating a convolutional layer
def Conv2D(
    inputs,
    params,
    use_pooling=False,
    activation_fn=tf.nn.relu,
    padding="SAME",
    name="Layer",
):
    """Builds a convolutional layer.
        Args:
            inputs: the inputs for convolution
            params: a dictionary with parameters for convolution
            use_pooling: if use pooling after convolution
            activation_fn: activation_fn after convloution
            paddding: type of padding 'SAME' or 'VALID'
            name: the name of the convolutuion
    """

    with tf.name_scope(name):
        weights = new_weights(params["weights"], name="weights")
        biases = new_biases(params["weights"][-1], name="biases")
        strides = params["strides"]

        # Do the convolution
        layer1 = (
            tf.nn.conv2d(inputs, filter=weights, strides=strides, padding=padding)
            + biases
        )

        if use_pooling:
            pool_kernel = params["pool_kernel"]
            pool_strides = params["pool_strides"]

            layer1 = tf.nn.avg_pool(
                value=layer1, ksize=pool_kernel, strides=pool_strides, padding=padding
            )

        return activation_fn(layer1)


class NMR_nn:
    """ Class for predicting the reactivty of reaction mixtures using
        experimental and theoretical spectrum."""

    def __init__(
        self, model_path="models/fullset", spectrum_shape=271, noise_level=0.5
    ):
        self.noise_level = noise_level
        self.spectrum_shape = spectrum_shape
        NUM_THREADS = 4
        self.config = tf.ConfigProto(intra_op_parallelism_threads=NUM_THREADS)
        self.session = tf.Session(config=self.config)
        self.initial_learning_rate = 5e-3
        self.num_epochs = 300
        self.batch_size = 8
        self.model_path = model_path

    def build_model(self, x, phase):
        """ Builds tensorflow graphs
            Args:
                inputs: an array of size [batch_size, 2, spectrum_shape]
                phase: a boolean if this is a training phase
            Returns:
                a tuple containg predicted logits and class labels
        """
        # Reshape to make it compatible with convolutioonal layers
        inputs = tf.reshape(x, shape=[-1, 2, self.spectrum_shape, 1])

        # Add noise to the spectra in the trainig phase
        # by multplying them by value from range 1-noise_level, 1+noise_level
        inputs_noisy = tf.cond(
            tf.equal(phase, tf.constant(True, tf.bool)),
            lambda: noisy(inputs, self.noise_level),
            lambda: no_noise(inputs),
        )

        params1a = {
            "weights": [2, 2, 1, 16],
            "strides": [1, 1, 1, 1],
            "pool_kernel": [1, 1, 2, 1],
            "pool_strides": [1, 1, 2, 1],
        }
        layer1a = Conv2D(inputs_noisy, params1a, use_pooling=True, name="Layer1a")

        params1b = {
            "weights": [2, 10, 1, 16],
            "strides": [1, 1, 1, 1],
            "pool_kernel": [1, 1, 2, 1],
            "pool_strides": [1, 1, 2, 1],
        }
        layer1b = Conv2D(inputs_noisy, params1b, use_pooling=True, name="Layer1b")

        params1c = {
            "weights": [2, 20, 1, 16],
            "strides": [1, 1, 1, 1],
            "pool_kernel": [1, 1, 2, 1],
            "pool_strides": [1, 1, 2, 1],
        }
        layer1c = Conv2D(inputs_noisy, params1c, use_pooling=True, name="Layer1c")

        layer1 = tf.concat([layer1a, layer1b, layer1c], axis=3)
        layer1 = tf.nn.local_response_normalization(layer1)

        params2 = {"weights": [1, 1, 48, 8], "strides": [1, 1, 1, 1]}
        layer2 = Conv2D(layer1, params2, name="Layer2")
        layer2_flat = tf.contrib.layers.flatten(layer2)
        layer4 = tf.layers.dense(layer2_flat, 512, activation=tf.nn.relu)
        dropout = tf.layers.dropout(inputs=layer4, rate=0.6)

        logits = tf.layers.dense(dropout, 4)
        y_pred = tf.nn.softmax(logits)

        return logits, y_pred


    def predict(self, datax):
        # Reset tensorflow graph
        tf.reset_default_graph()

        # Placeholders for feeding the data
        x = tf.placeholder(tf.float32, shape=[None, 2, self.spectrum_shape])
        phase = tf.placeholder(tf.bool)

        # Build the model and get logits and predicted reactivty classes
        logits, y_pred = self.build_model(x, phase)

        sess = tf.Session()
        saver = tf.train.Saver()
        sess.run(tf.global_variables_initializer())
        saver.restore(sess, self.model_path)

        feed_dict = {x: datax, phase: False}
        prediction = sess.run([y_pred], feed_dict=feed_dict)
        return prediction


def full_nmr_process(file_path, reagents, reagent_folder):
    """

    :param file_path: folder of raw nmr data
    :param reagents: list of reagent names, these are the folder names in reagent_folder
    :param reagent_folder: the folder cotaining the nmr of the reagents.
    :return: value of rectivity between 0 and 1
    """
    data_x = make_input_matrix(file_path, reagents, reagent_folder)
    nn = NMR_nn()
    reactivity = nn.predict(data_x)
    weighted_reactivity = reactivity * (np.array([0, 1, 2, 3]))
    return np.sum(weighted_reactivity) / 3


if __name__ == "__main__":
    file_path = "Z:\\group\\Dario Caramelli\\Projects\\FinderX\\data\\20180418-1809-photochemical_space\\0013\\0013-post-reactor2-NMR-20180419-0545"
    reagents = ["phenylhydrazine", "glycidyl_propargyl_ether"]
    reagent_folder = "Z:\\group\\Dario Caramelli\\Projects\\FinderX\\data\\002_photo_space\\reagents\\"

    print(full_nmr_process(file_path, reagents, reagent_folder))

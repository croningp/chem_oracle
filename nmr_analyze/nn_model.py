import tensorflow as tf
import math
import numpy as np
from nmr_analyze.utils import *
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

    def train(self, trsetx, trsety, valsetx, valsety):

        tf.reset_default_graph()

        x = tf.placeholder(tf.float32, shape=[None, 2, self.spectrum_shape])
        phase = tf.placeholder(tf.bool)
        y_true = tf.placeholder(tf.float32, shape=[None, 4])

        logits, y_pred = self.build_model(x, phase)

        cross_entropy = -(
            y_true * tf.log(y_pred + 1e-12) + (1 - y_true) * tf.log(1 - y_pred + 1e-12)
        )
        loss = tf.reduce_mean(cross_entropy)

        # Define global step and decaying learning rate
        global_step = tf.Variable(0, trainable=False)
        learning_rate = tf.train.exponential_decay(
            learning_rate=self.initial_learning_rate,
            global_step=global_step,
            decay_steps=300,
            decay_rate=0.95,
            staircase=False,
        )
        # Define optimization step
        optimize = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(
            loss, global_step=global_step
        )

        sess = tf.Session()
        saver = tf.train.Saver()
        sess.run(tf.global_variables_initializer())
        training_data_manager = DataManager(trsetx, trsety, random_shift=True)
        iter_per_epoch = math.ceil(len(trsetx) / self.batch_size)

        best_val_loss = np.inf
        val_losses = []
        training_losses = []
        epochs = []

        for e in range(self.num_epochs):
            e_losses = []
            for i in range(iter_per_epoch):
                batchx, batchy = training_data_manager.next_batch(self.batch_size)
                # batchy = batchy.reshape([-1, 1])
                _, iloss = sess.run(
                    [optimize, loss], feed_dict={x: batchx, y_true: batchy, phase: True}
                )
                #                 print ('loss:\t ', iloss)
                e_losses.append(iloss)
            print("Epoch {} training loss {}".format(e, np.mean(e_losses)))

            val_loss = sess.run(
                loss, feed_dict={x: valsetx, y_true: valsety, phase: False}
            )

            val_losses.append(val_loss)
            training_losses.append(np.mean(e_losses))
            epochs.append(e)

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                print("New validation loss found {}".format(val_loss))
                saver.save(sess, self.model_path)

    def test(self, testx, testy):
        # Reset tensorflow graph
        tf.reset_default_graph()

        # Placeholders for feeding the data
        x = tf.placeholder(tf.float32, shape=[None, 2, self.spectrum_shape])
        phase = tf.placeholder(tf.bool)
        y_true = tf.placeholder(tf.float32, shape=[None, 4])

        # Build the model and get logits and predicted reactivty classes
        logits, y_pred = self.build_model(x, phase)
        # Define crossentropy loss here
        cross_entropy = -(
            y_true * tf.log(y_pred + 1e-12) + (1 - y_true) * tf.log(1 - y_pred + 1e-12)
        )
        loss = tf.reduce_mean(cross_entropy)

        top2_accuracy = tf.reduce_mean(
            tf.cast(
                tf.nn.in_top_k(
                    predictions=y_pred, targets=tf.argmax(y_true, axis=-1), k=2
                ),
                tf.float32,
            )
        )
        accuracy = tf.reduce_mean(
            tf.cast(
                tf.nn.in_top_k(
                    predictions=y_pred, targets=tf.argmax(y_true, axis=-1), k=1
                ),
                tf.float32,
            )
        )

        sess = tf.Session()
        saver = tf.train.Saver()
        sess.run(tf.global_variables_initializer())
        saver.restore(sess, self.model_path)

        feed_dict = {x: testx, y_true: testy, phase: False}
        test_loss, test_pred, test_accuracy, test_top2_accuracy = sess.run(
            [loss, y_pred, accuracy, top2_accuracy], feed_dict=feed_dict
        )
        reactivity = np.argmax(test_pred, axis=1)

        print("--------------------------Test accuracy {}".format(test_accuracy))
        print("Top 2 accuracy {}".format(test_top2_accuracy))

        true_reactivity = np.argmax(testy, axis=1)
        hits = np.equal(reactivity, true_reactivity)
        accuracy = np.mean(hits.astype(np.float32))

        results = [[i, j] for i, j in zip(reactivity, true_reactivity)]

        distr = [[], [], [], []]

        result = 0

        for exp in results:
            distr[exp[1]].append(exp)
        for i in range(len(distr)):
            count = 0
            clas = distr[i]
            for exp in clas:
                if exp[1] == exp[0]:
                    count += 1
            print("{} - accuracy of {}%".format(i, count / len(clas)))
        result += count / len(clas)
        m = np.zeros([4, 4])
        for i in results:
            m[i[0], i[1]] += 1
        # %matplotlib qt
        plt.figure()
        plt.imshow(m)
        #         plt.title(param)
        plt.xlabel("True", fontsize=15)
        plt.ylabel("Predicted", fontsize=15)
        for y in range(4):
            for x in range(4):
                plt.annotate(int(m[y, x]), (x - 0.2, y + 0.1), fontsize=17)
        print(reactivity)

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
        reactivity = np.argmax(prediction, axis=1)
        return prediction


def full_nmr_process(file_path, reagents):
    """

    :param file_path: folder of raw nmr data
    :param reagents: list of reagent names, the reagents folder is currently hardcoded inside the get_theoretical_nmr
    in utils.
    :return: value of rectivity between 0 and 1
    """
    data = raw_nmr_to_dataframe(file_path, reagents)
    data_x, data_y = read_data(data)
    nn = NMR_nn()
    reactivity = nn.predict(data_x)
    weighted_reactivity = reactivity * (np.array([0, 1, 2, 3]))
    return np.sum(weighted_reactivity) / 3


if __name__ == "__main__":
    file_path = "Z:\\group\\Dario Caramelli\\Projects\\FinderX\\data\\20180418-1809-photochemical_space\\0013\\0013-post-reactor2-NMR-20180419-0545"
    reagents = ["phenylhydrazine", "glycidyl_propargyl_ether"]

    print(full_nmr_process(file_path, reagents))

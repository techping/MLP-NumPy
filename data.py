# data file
# load data from tf API
import tensorflow as tf

def get_data(dataset='fmnist'):
    if dataset == 'fmnist':
        # Load the fashion-mnist pre-shuffled train data and test data
        (x_train, y_train), (x_test, y_test) = tf.keras.datasets.fashion_mnist.load_data()
        x_train = x_train.reshape(-1, 28 * 28).astype('float32') / 255.
        y_train = tf.keras.utils.to_categorical(y_train, 10)
        x_test = x_test.reshape(-1, 28 * 28).astype('float32') / 255.
        y_test = tf.keras.utils.to_categorical(y_test, 10)
    elif dataset == 'mnist':
        (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
        x_train = x_train.reshape(-1, 28 * 28).astype('float32') / 255.
        y_train = tf.keras.utils.to_categorical(y_train, 10)
        x_test = x_test.reshape(-1, 28 * 28).astype('float32') / 255.
        y_test = tf.keras.utils.to_categorical(y_test, 10)
    else:
        # TODO:
        # add other datasets
        pass
    print("x_train shape:", x_train.shape, "y_train shape:", y_train.shape)
    return x_train, y_train, x_test, y_test
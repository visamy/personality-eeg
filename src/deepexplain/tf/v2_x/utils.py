import numpy as np
import tensorflow as tf
import warnings
from src.deepexplain.tf.v2_x import constants
from tensorflow.python.ops import nn_grad, math_grad


# -----------------------------------------------------------------------------
# UTILITY FUNCTIONS
# -----------------------------------------------------------------------------
# Some of the following functions for batch processing have been borrowed and adapter from Keras
# https://github.com/keras-team/keras/blob/master/keras/utils/generic_utils.py
# https://github.com/keras-team/keras/blob/master/keras/engine/training_utils.py


def make_batches(size, batch_size):
    """Returns a list of batch indices (tuples of indices).
    # Arguments
        size: Integer, total size of the data to slice into batches.
        batch_size: Integer, batch size.
    # Returns
        A list of tuples of array indices.
    """
    num_batches = (size + batch_size - 1) // batch_size  # round up
    return [(i * batch_size, min(size, (i + 1) * batch_size))
            for i in range(num_batches)]


def to_list(x, allow_tuple=False):
    """Normalizes a list/tensor into a list.
    If a tensor is passed, we return
    a list of size 1 containing the tensor.
    # Arguments
        x: target object to be normalized.
        allow_tuple: If False and x is a tuple,
            it will be converted into a list
            with a single element (the tuple).
            Else converts the tuple to a list.
    # Returns
        A list.
    """
    if isinstance(x, list):
        return x
    if allow_tuple and isinstance(x, tuple):
        return list(x)
    return [x]


def unpack_singleton(x):
    """Gets the equivalent np-array if the iterable has only one value.
    Otherwise return the iterable.
    # Argument
        x: A list or tuple.
    # Returns
        The same iterable or the iterable converted to a np-array.
    """
    if len(x) == 1:
        return np.array(x)
    return x


def slice_arrays(arrays, start=None, stop=None):
    """Slices an array or list of arrays.
    """
    if arrays is None:
        return [None]
    elif isinstance(arrays, list):
        return [None if x is None else x[start:stop] for x in arrays]
    else:
        return arrays[start:stop]


def placeholder_from_data(numpy_array):
    if numpy_array is None:
        return None
    return tf.compat.v1.placeholder('float', [None,] + list(numpy_array.shape[1:]))

def original_grad(op, grad):
    """
    Return original Tensorflow gradient for an op
    :param op: op
    :param grad: Tensor
    :return: Tensor
    """
    if op.type not in constants.SUPPORTED_ACTIVATIONS:
        warnings.warn('Activation function (%s) not supported' % op.type)
    opname = '_%sGrad' % op.type
    if hasattr(nn_grad, opname):
        f = getattr(nn_grad, opname)
    else:
        f = getattr(math_grad, opname)
    return f(op, grad)

def activation(type):
    """
    Returns Tensorflow's activation op, given its type
    :param type: string
    :return: op
    """
    if type not in constants.SUPPORTED_ACTIVATIONS:
        warnings.warn('Activation function (%s) not supported' % type)
    f = getattr(tf.nn, type.lower())
    return f
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import numpy as np
import warnings, logging

from deepexplain.tf.v2_x.utils import make_batches, slice_arrays, to_list, unpack_singleton, placeholder_from_data, original_grad
# -----------------------------------------------------------------------------
# ATTRIBUTION METHODS BASE CLASSES
# -----------------------------------------------------------------------------


class AttributionMethod(object):
    """
    Attribution method base class
    """
    def __init__(self, T, X, session, keras_learning_phase=None, Y_shape=None):
        self.T = T  # target Tensor
        self.X = X  # input Tensor
        self.Y_shape = Y_shape if Y_shape is not None else [None,] + T.get_shape().as_list()[1:]
        # Most often T contains multiple output units. In this case, it is often necessary to select
        # a single unit to compute contributions for. This can be achieved passing 'ys' as weight for the output Tensor.
        self.Y = tf.compat.v1.placeholder(tf.float32, self.Y_shape)
        # placeholder_from_data(ys) if ys is not None else 1.0  # Tensor that represents weights for T
        self.T = self.T * self.Y
        self.symbolic_attribution = None
        self.session = session
        self.keras_learning_phase = keras_learning_phase
        self.has_multiple_inputs = type(self.X) is list or type(self.X) is tuple
        logging.info('Model with multiple inputs: %s' % self.has_multiple_inputs)

        # Set baseline
        # TODO: now this sets a baseline also for those methods that does not require it
        self._set_check_baseline()

        # References
        self._init_references()

        # Create symbolic explanation once during construction (affects only gradient-based methods)
        self.explain_symbolic()

    def explain_symbolic(self):
        return None

    def run(self, xs, ys=None, batch_size=None):
        pass

    def _init_references(self):
        pass

    def _check_input_compatibility(self, xs, ys=None, batch_size=None):
        if ys is not None and len(ys) != len(xs):
            raise RuntimeError('When provided, the number of elements in ys must equal the number of elements in xs')
        if batch_size is not None and batch_size > 0:
            if self.T.shape[0] is not None and self.T.shape[0] is not batch_size:
                raise RuntimeError('When using batch evaluation, the first dimension of the target tensor '
                                    'must be compatible with the batch size. Found %s instead' % self.T.shape[0])
            if isinstance(self.X, list):
                for x in self.X:
                    if x.shape[0] is not None and x.shape[0] is not batch_size:
                        raise RuntimeError('When using batch evaluation, the first dimension of the input tensor '
                                            'must be compatible with the batch size. Found %s instead' % x.shape[
                                                0])
            else:
                if self.X.shape[0] is not None and self.X.shape[0] is not batch_size:
                    raise RuntimeError('When using batch evaluation, the first dimension of the input tensor '
                                        'must be compatible with the batch size. Found %s instead' % self.X.shape[0])

    def _session_run_batch(self, T, xs, ys=None):
        feed_dict = {}
        if self.has_multiple_inputs:
            for k, v in zip(self.X, xs):
                feed_dict[k] = v
        else:
            feed_dict[self.X] = xs

        # If ys is not passed, produce a vector of ones that will be broadcasted to all batch samples
        feed_dict[self.Y] = ys if ys is not None else np.ones([1,] + self.Y_shape[1:])

        if self.keras_learning_phase is not None:
            feed_dict[self.keras_learning_phase] = 0
        return self.session.run(T, feed_dict)

    def _session_run(self, T, xs, ys=None, batch_size=None):
        num_samples = len(xs)
        if self.has_multiple_inputs is True:
            num_samples = len(xs[0])
            if len(xs) != len(self.X):
                raise RuntimeError('List of input tensors and input data have different lengths (%s and %s)'
                                    % (str(len(xs)), str(len(self.X))))
            if batch_size is not None:
                for xi in xs:
                    if len(xi) != num_samples:
                        raise RuntimeError('Evaluation in batches requires all inputs to have '
                                            'the same number of samples')

        if batch_size is None or batch_size <= 0 or num_samples <= batch_size:
            return self._session_run_batch(T, xs, ys)
        else:
            outs = []
            batches = make_batches(num_samples, batch_size)
            for batch_index, (batch_start, batch_end) in enumerate(batches):
                # Get a batch from data
                xs_batch = slice_arrays(xs, batch_start, batch_end)
                # If the target tensor has one entry for each sample, we need to batch it as well
                ys_batch = None
                if ys is not None:
                    ys_batch = slice_arrays(ys, batch_start, batch_end)
                batch_outs = self._session_run_batch(T, xs_batch, ys_batch)
                batch_outs = to_list(batch_outs)
                if batch_index == 0:
                    # Pre-allocate the results arrays.
                    for batch_out in batch_outs:
                        shape = (num_samples,) + batch_out.shape[1:]
                        outs.append(np.zeros(shape, dtype=batch_out.dtype))
                for i, batch_out in enumerate(batch_outs):
                    outs[i][batch_start:batch_end] = batch_out
            return unpack_singleton(outs)

    @staticmethod
    def compare_shape(l1, l2):
        if len(l1) != len(l2):
            return False
        else:
            for elm1, elm2 in zip(l1, l2):
                if elm1 is None or elm2 is None: # None is like a place holder
                    continue
                elif elm1 != elm2:
                    return False
            return True

    def _set_check_baseline(self):
        # Do nothing for those methods that have no baseline required
        if not hasattr(self, "baseline"):
            return

        if self.baseline is None:
            if self.has_multiple_inputs:
                self.baseline = [np.zeros([1,] + xi.get_shape().as_list()[1:]) for xi in self.X]
            else:
                self.baseline = np.zeros([1,] + self.X.get_shape().as_list()[1:])

        else:
            if self.has_multiple_inputs:
                for i, xi in enumerate(self.X):
                    if AttributionMethod.compare_shape(list(self.baseline[i].shape), xi.get_shape().as_list()[1:]):
                        self.baseline[i] = np.expand_dims(self.baseline[i], 0)
                    else:
                        raise RuntimeError('Baseline shape %s does not match expected shape %s'
                                            % (self.baseline[i].shape, self.X.get_shape().as_list()[1:]))
            else:
                if AttributionMethod.compare_shape(self.baseline.shape, self.X.get_shape().as_list()[1:]):
                    self.baseline = np.expand_dims(self.baseline, 0)
                else:
                    raise RuntimeError('Baseline shape %s does not match expected shape %s'
                                        % (self.baseline.shape, self.X.get_shape().as_list()[1:]))

class GradientBasedMethod(AttributionMethod):
    """
    Base class for gradient-based attribution methods
    """
    def get_symbolic_attribution(self):
        return tf.gradients(ys=self.T, xs=self.X)

    def explain_symbolic(self):
        if self.symbolic_attribution is None:
            self.symbolic_attribution = self.get_symbolic_attribution()
        return self.symbolic_attribution

    def run(self, xs, ys=None, batch_size=None):
        self._check_input_compatibility(xs, ys, batch_size)
        results = self._session_run(self.explain_symbolic(), xs, ys, batch_size)
        return results[0] if not self.has_multiple_inputs else results

    @classmethod
    def nonlinearity_grad_override(cls, op, grad):
        return original_grad(op, grad)


class PerturbationBasedMethod(AttributionMethod):
    """
        Base class for perturbation-based attribution methods
    """
    def __init__(self, T, X, session, keras_learning_phase, Y_shape):
        super(PerturbationBasedMethod, self).__init__(T, X, session, keras_learning_phase, Y_shape)
        self.base_activation = None
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys
import numpy as np
from skimage.util import view_as_windows
import warnings
import logging
import tensorflow as tf
from tqdm import tqdm

from tensorflow.python.ops import nn_grad, math_grad

from src.deepexplain.tf.v2_x.utils import make_batches, slice_arrays, to_list, unpack_singleton, placeholder_from_data, original_grad, activation
from src.deepexplain.tf.v2_x.baseClasses import GradientBasedMethod, PerturbationBasedMethod
from src.deepexplain.tf.v2_x import constants


# -----------------------------------------------------------------------------
# ATTRIBUTION METHODS
# -----------------------------------------------------------------------------
"""
Returns zero attributions. For testing only.
"""


class DummyZero(GradientBasedMethod):
    def get_symbolic_attribution(self,):
        return tf.gradients(ys=self.T, xs=self.X)

    @classmethod
    def nonlinearity_grad_override(cls, op, grad):
        input = op.inputs[0]
        return tf.zeros_like(input)


"""
Saliency maps
https://arxiv.org/abs/1312.6034
"""


class Saliency(GradientBasedMethod):
    def get_symbolic_attribution(self):
        return [tf.abs(g) for g in tf.gradients(ys=self.T, xs=self.X)]


"""
Gradient * Input
https://arxiv.org/pdf/1704.02685.pdf - https://arxiv.org/abs/1611.07270
"""


class GradientXInput(GradientBasedMethod):
    def get_symbolic_attribution(self):
        return [g * x for g, x in zip(
            tf.gradients(ys=self.T, xs=self.X),
            self.X if self.has_multiple_inputs else [self.X])]

"""
Layer-wise Relevance Propagation with epsilon rule
http://journals.plos.org/plosone/article?id=10.1371/journal.pone.0130140
"""


class EpsilonLRP(GradientBasedMethod):
    eps = None

    def __init__(self, T, X, session, keras_learning_phase, epsilon=1e-4, Y_shape=None):
        assert epsilon > 0.0, 'LRP epsilon must be greater than zero'
        global eps
        eps = epsilon
        super(EpsilonLRP, self).__init__(T, X, session, keras_learning_phase, Y_shape)

    def get_symbolic_attribution(self):
        return [g * x for g, x in zip(
            tf.gradients(ys=self.T, xs=self.X),
            self.X if self.has_multiple_inputs else [self.X])]

    @classmethod
    def nonlinearity_grad_override(cls, op, grad):
        output = op.outputs[0]
        input = op.inputs[0]
        return grad * output / (input + eps *
                                tf.compat.v1.where(input >= 0, tf.ones_like(input), -1 * tf.ones_like(input)))


"""
Integrated Gradients
https://arxiv.org/pdf/1703.01365.pdf
"""
class IntegratedGradients(GradientBasedMethod):
    def __init__(self, T, X, session, keras_learning_phase, steps=100, baseline=None, Y_shape=None):
        self.steps = steps
        self.baseline = baseline
        super(IntegratedGradients, self).__init__(T, X, session, keras_learning_phase, Y_shape)

    def run(self, xs, ys=None, batch_size=None):
        self._check_input_compatibility(xs, ys, batch_size)

        gradient = None
        for alpha in tqdm(list(np.linspace(1. / self.steps, 1.0, self.steps))):
            xs_mod = [b + (x - b) * alpha for x, b in zip(xs, self.baseline)] if self.has_multiple_inputs \
                else self.baseline + (xs - self.baseline) * alpha
            _attr = self._session_run(self.explain_symbolic(), xs_mod, ys, batch_size)
            if gradient is None: gradient = _attr
            else: gradient = [g + a for g, a in zip(gradient, _attr)]

        results = [g * (x - b) / self.steps for g, x, b in zip(
            gradient,
            xs if self.has_multiple_inputs else [xs],
            self.baseline if self.has_multiple_inputs else [self.baseline])]

        return results[0] if not self.has_multiple_inputs else results



"""
DeepLIFT
This reformulation only considers the "Rescale" rule
https://arxiv.org/abs/1704.02685
"""
class DeepLIFTRescale(GradientBasedMethod):

    _deeplift_ref = {}

    def __init__(self, T, X, session, keras_learning_phase, baseline=None, Y_shape=None):
        self.baseline = baseline
        super(DeepLIFTRescale, self).__init__(T, X, session, keras_learning_phase, Y_shape)

    def get_symbolic_attribution(self):
        return [g * (x - b) for g, x, b in zip(
            tf.gradients(ys=self.T, xs=self.X),
            self.X if self.has_multiple_inputs else [self.X],
            self.baseline if self.has_multiple_inputs else [self.baseline])]

    @classmethod
    def nonlinearity_grad_override(cls, op, grad):
        output = op.outputs[0]
        input = op.inputs[0]
        ref_input = cls._deeplift_ref[op.name]
        ref_output = activation(op.type)(ref_input)
        delta_out = output - ref_output
        delta_in = input - ref_input
        instant_grad = activation(op.type)(0.5 * (ref_input + input))
        return tf.compat.v1.where(tf.abs(delta_in) > 1e-5, grad * delta_out / delta_in,
                        original_grad(instant_grad.op, grad))

    def _init_references(self):
        # print ('DeepLIFT: computing references...')
        sys.stdout.flush()
        self._deeplift_ref.clear()
        ops = []
        g = tf.compat.v1.get_default_graph()
        for op in g.get_operations():
            if len(op.inputs) > 0 and not op.name.startswith('gradients'):
                if op.type in constants.SUPPORTED_ACTIVATIONS:
                    ops.append(op)
        YR = self._session_run([o.inputs[0] for o in ops], self.baseline)
        for (r, op) in zip(YR, ops):
            self._deeplift_ref[op.name] = r
        # print('DeepLIFT: references ready')
        sys.stdout.flush()


"""
Occlusion method
Generalization of the grey-box method presented in https://arxiv.org/pdf/1311.2901.pdf
This method performs a systematic perturbation of contiguous hyperpatches in the input,
replacing each patch with a user-defined value (by default 0).

window_shape : integer or tuple of length xs_ndim
Defines the shape of the elementary n-dimensional orthotope the rolling window view.
If an integer is given, the shape will be a hypercube of sidelength given by its value.

step : integer or tuple of length xs_ndim
Indicates step size at which extraction shall be performed.
If integer is given, then the step is uniform in all dimensions.
"""
class Occlusion(PerturbationBasedMethod):

    def __init__(self, T, X, session, keras_learning_phase, window_shape=None, step=None):
        super(Occlusion, self).__init__(T, X, session, keras_learning_phase)
        if self.has_multiple_inputs:
            raise RuntimeError('Multiple inputs not yet supported for perturbation methods')

        input_shape = X[0].get_shape().as_list()
        if window_shape is not None:
            assert len(window_shape) == len(input_shape), \
                'window_shape must have length of input (%d)' % len(input_shape)
            self.window_shape = tuple(window_shape)
        else:
            self.window_shape = (1,) * len(input_shape)

        if step is not None:
            assert isinstance(step, int) or len(step) == len(input_shape), \
                'step must be integer or tuple with the length of input (%d)' % len(input_shape)
            self.step = step
        else:
            self.step = 1
        self.replace_value = 0.0
        logging.info('Input shape: %s; window_shape %s; step %s' % (input_shape, self.window_shape, self.step))

    def run(self, xs, ys=None, batch_size=None):
        self._check_input_compatibility(xs, ys, batch_size)
        input_shape = xs.shape[1:]
        batch_size = xs.shape[0]
        total_dim = np.prod(input_shape).item()

        # Create mask
        index_matrix = np.arange(total_dim).reshape(input_shape)
        idx_patches = view_as_windows(index_matrix, self.window_shape, self.step).reshape((-1,) + self.window_shape)
        heatmap = np.zeros_like(xs, dtype=np.float32).reshape((-1), total_dim)
        w = np.zeros_like(heatmap)

        # Compute original output
        eval0 = self._session_run(self.T, xs, ys, batch_size)

        # Start perturbation loop
        for i, p in enumerate(tqdm(idx_patches)):
            mask = np.ones(input_shape).flatten()
            mask[p.flatten()] = self.replace_value
            masked_xs = mask.reshape((1,) + input_shape) * xs
            delta = eval0 - self._session_run(self.T, masked_xs, ys, batch_size)
            delta_aggregated = np.sum(delta.reshape((batch_size, -1)), -1, keepdims=True)
            heatmap[:, p.flatten()] += delta_aggregated
            w[:, p.flatten()] += p.size

        attribution = np.reshape(heatmap / w, xs.shape)
        if np.isnan(attribution).any():
            warnings.warn('Attributions generated by Occlusion method contain nans, '
                            'probably because window_shape and step do not allow to cover the all input.')
        return attribution


"""
Shapley Value sampling
Computes approximate Shapley Values using "Polynomial calculation of the Shapley value based on sampling",
Castro et al, 2009 (https://www.sciencedirect.com/science/article/pii/S0305054808000804)

samples : integer (default 5)
Defined the number of samples for each input feature.
Notice that evaluating a model samples * n_input_feature times might take a while.

sampling_dims : list of dimension indexes to run sampling on (feature dimensions).
By default, all dimensions except the batch dimension will be sampled.
For example, with a 4-D tensor that contains color images, single color channels are sampled.
To sample pixels, instead, use sampling_dims=[1,2]
"""
class ShapleySampling(PerturbationBasedMethod):

    def __init__(self, T, X, session, keras_learning_phase, samples=5, sampling_dims=None, Y_shape=None):
        super(ShapleySampling, self).__init__(T, X, session, keras_learning_phase, Y_shape)
        if self.has_multiple_inputs:
            raise RuntimeError('Multiple inputs not yet supported for perturbation methods')
        dims = len(X.shape)
        if sampling_dims is not None:
            if not 0 < len(sampling_dims) <= (dims - 1):
                raise RuntimeError('sampling_dims must be a list containing 1 to %d elements' % (dims-1))
            if 0 in sampling_dims:
                raise RuntimeError('Cannot sample batch dimension: remove 0 from sampling_dims')
            if any([x < 1 or x > dims-1 for x in sampling_dims]):
                raise RuntimeError('Invalid value in sampling_dims')
        else:
            sampling_dims = list(range(1, dims))

        self.samples = samples
        self.sampling_dims = sampling_dims

    def run(self, xs, ys=None, batch_size=None):
        xs_shape = list(xs.shape)
        batch_size = xs.shape[0]
        n_features = int(np.prod([xs.shape[i] for i in self.sampling_dims]).item())
        result = np.zeros((xs_shape[0], n_features))

        run_shape = list(xs_shape)  # a copy
        run_shape = np.delete(run_shape, self.sampling_dims).tolist()
        run_shape.insert(1, -1)

        reconstruction_shape = [xs_shape[0]]
        for j in self.sampling_dims:
            reconstruction_shape.append(xs_shape[j])
        with tqdm(total=self.samples * n_features) as pbar:
            for _ in range(self.samples):
                p = np.random.permutation(n_features)
                x = xs.copy().reshape(run_shape)
                y = None
                for i in p:
                    if y is None:
                        y = self._session_run(self.T, x.reshape(xs_shape), ys, batch_size)
                    x[:, i] = 0
                    y0 = self._session_run(self.T, x.reshape(xs_shape), ys, batch_size)
                    delta = y - y0
                    delta_aggregated = np.sum(delta.reshape((batch_size, -1)), -1, keepdims=False)
                    result[:, i] += delta_aggregated
                    y = y0
                pbar.update(1)

        shapley = result / self.samples
        return shapley.reshape(reconstruction_shape)

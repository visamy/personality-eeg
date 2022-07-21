from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
from tensorflow.python.framework import ops
from collections import OrderedDict
import warnings, logging

from src.deepexplain.tf.v1_x import constants
from src.deepexplain.tf.v1_x.baseClasses import GradientBasedMethod
from src.deepexplain.tf.v1_x.methods import DeepLIFTRescale, EpsilonLRP
from src.deepexplain.tf.v1_x.utils import original_grad

from src.deepexplain.tf.v1_x.methods import DummyZero, Saliency, GradientXInput, IntegratedGradients, EpsilonLRP, DeepLIFTRescale, Occlusion, ShapleySampling
attribution_methods = OrderedDict({
    'zero': (DummyZero, 0),
    'saliency': (Saliency, 1),
    'grad*input': (GradientXInput, 2),
    'intgrad': (IntegratedGradients, 3),
    'elrp': (EpsilonLRP, 4),
    'deeplift': (DeepLIFTRescale, 5),
    'occlusion': (Occlusion, 6),
    'shapley_sampling': (ShapleySampling, 7)
})
print(f'Using tf version = {tf.__version__}')


@ops.RegisterGradient("DeepExplainGrad")
def deepexplain_grad(op, grad):
    # constants._ENABLED_METHOD_CLASS, _GRAD_OVERRIDE_CHECKFLAG
    constants._GRAD_OVERRIDE_CHECKFLAG = 1
    if constants._ENABLED_METHOD_CLASS is not None \
            and issubclass(constants._ENABLED_METHOD_CLASS, GradientBasedMethod):
        return constants._ENABLED_METHOD_CLASS.nonlinearity_grad_override(op, grad)
    else:
        return original_grad(op, grad)


class DeepExplain(object):
    def __init__(self, graph=None, session=tf.compat.v1.get_default_session()):
        self.method = None
        self.batch_size = None
        self.session = session
        self.graph = session.graph if graph is None else graph
        self.graph_context = self.graph.as_default()
        self.override_context = self.graph.gradient_override_map(self.get_override_map())
        self.keras_phase_placeholder = None
        self.context_on = False
        if self.session is None:
            raise RuntimeError('DeepExplain: could not retrieve a session. Use DeepExplain(session=your_session).')

    def __enter__(self):
        # Override gradient of all ops created in context
        self.graph_context.__enter__()
        self.override_context.__enter__()
        self.context_on = True
        return self

    def __exit__(self, type, value, traceback):
        self.graph_context.__exit__(type, value, traceback)
        self.override_context.__exit__(type, value, traceback)
        self.context_on = False

    def get_explainer(self, method, T, X, **kwargs):
        if not self.context_on:
            raise RuntimeError('Explain can be called only within a DeepExplain context.')
        # global constants._ENABLED_METHOD_CLASS, _GRAD_OVERRIDE_CHECKFLAG
        self.method = method
        if self.method in attribution_methods:
            method_class, method_flag = attribution_methods[self.method]
        else:
            raise RuntimeError('Method must be in %s' % list(attribution_methods.keys()))
        if isinstance(X, list):
            for x in X:
                if 'tensor' not in str(type(x)).lower():
                    raise RuntimeError('If a list, X must contain only Tensorflow Tensor objects')
        else:
            if 'tensor' not in str(type(X)).lower():
                raise RuntimeError('X must be a Tensorflow Tensor object or a list of them')

        if 'tensor' not in str(type(T)).lower():
            raise RuntimeError('T must be a Tensorflow Tensor object')
        # logging.info('DeepExplain: running "%s" explanation method (%d)' % (self.method, method_flag))
        self._check_ops()
        constants._GRAD_OVERRIDE_CHECKFLAG = 0

        constants._ENABLED_METHOD_CLASS = method_class
        method = constants._ENABLED_METHOD_CLASS(T, X,
                                        self.session,
                                        keras_learning_phase=self.keras_phase_placeholder,
                                       **kwargs)

        if (issubclass(constants._ENABLED_METHOD_CLASS, DeepLIFTRescale) or issubclass(constants._ENABLED_METHOD_CLASS, EpsilonLRP)) \
            and constants._GRAD_OVERRIDE_CHECKFLAG == 0:
            warnings.warn('DeepExplain detected you are trying to use an attribution method that requires '
                            'gradient override but the original gradient was used instead. You might have forgot to '
                            '(re)create your graph within the DeepExlain context. Results are not reliable!')
        constants._ENABLED_METHOD_CLASS = None
        constants._GRAD_OVERRIDE_CHECKFLAG = 0
        self.keras_phase_placeholder = None
        return method

    def explain(self, method, T, X, xs, ys=None, batch_size=None, **kwargs):
        explainer = self.get_explainer(method, T, X, **kwargs)
        return explainer.run(xs, ys, batch_size)

    @staticmethod
    def get_override_map():
        return dict((a, 'DeepExplainGrad') for a in constants.SUPPORTED_ACTIVATIONS)

    def _check_ops(self):
        """
        Heuristically check if any op is in the list of unsupported activation functions.
        This does not cover all cases where explanation methods would fail, and must be improved in the future.
        Also, check if the placeholder named 'keras_learning_phase' exists in the graph. This is used by Keras
        and needs to be passed in feed_dict.
        :return:
        """
        g = tf.compat.v1.get_default_graph()
        for op in g.get_operations():
            if len(op.inputs) > 0 and not op.name.startswith('gradients'):
                if op.type in constants.UNSUPPORTED_ACTIVATIONS:
                    warnings.warn('Detected unsupported activation (%s). '
                                    'This might lead to unexpected or wrong results.' % op.type)
            elif 'keras_learning_phase' in op.name:
                self.keras_phase_placeholder = op.outputs[0]
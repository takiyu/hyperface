# -*- coding: utf-8 -*-

import six

from chainer import reporter as reporter_module
from chainer.training import extensions
from chainer import variable


class SequentialEvaluator(extensions.Evaluator):
    '''  Sequential Evaluator
    The differences from original code are
        * One batch evaluation (no shallow copy)
        * Removal of `summary`.
    '''

    def evaluate(self):
        iterator = self._iterators['main']
        target = self._targets['main']
        eval_func = self.eval_func or target

        if self.eval_hook:
            self.eval_hook(self)

        batch = next(iterator)

        observation = {}
        with reporter_module.report_scope(observation):
            in_arrays = self.converter(batch, self.device)
            if isinstance(in_arrays, tuple):
                in_vars = tuple(variable.Variable(x, volatile='on')
                                for x in in_arrays)
                eval_func(*in_vars)
            elif isinstance(in_arrays, dict):
                in_vars = {key: variable.Variable(x, volatile='on')
                           for key, x in six.iteritems(in_arrays)}
                eval_func(**in_vars)
            else:
                in_var = variable.Variable(in_arrays, volatile='on')
                eval_func(in_var)

        return observation

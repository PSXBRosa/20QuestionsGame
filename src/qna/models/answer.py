from collections import namedtuple
import numpy as np

class Answer(namedtuple('Answer', ['status', 'prob', 'word', 'word_vec_repr'])):
    def __or__(self, other):
        return (2 * self.status - 1) * np.dot(self.word_vec_repr, other)

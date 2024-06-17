import numpy as np
from numba import jit
from ._base import BaseAnswerModel


@jit(nopython=True)
def find_closest(item, vec):
    closest = (-1, float('inf'))
    for i, val in enumerate(vec):
        if val == item:
            return i, val
        dist = abs(item - val)
        if dist < closest[1]:
            closest = (i, dist)
    return closest


class VectorSimilarityAnswerModel(BaseAnswerModel):
    def get_highest_entropy_doc_index(self, vectorspace, probdist):
        doc_similarities = vectorspace.get_similarity_matrix()
        prob_word_is_rel = np.sum(
                np.abs(doc_similarities) * probdist[:, None],
                axis=0
        )  # multiply the rows by the associated prob and then sum the columns
        # for binary outcomes, the highest entropy distribution is a 50/50
        # chance. So we'll select word for which the answer to the question
        # 'is the answer related to {word}' has an equal probability of be-
        # ing yes and no.
        return find_closest(0.5, prob_word_is_rel)


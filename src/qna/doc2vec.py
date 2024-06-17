import logging
from collections.abc import Iterable
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from nltk.tokenize import word_tokenize
import numpy as np
from .models import BaseAnswerModel, Answer

LOGGER = logging.getLogger(__name__)

class DocVectorSpace:
    def __init__(self, data: dict, **kwargs):
        LOGGER.debug("Initializing a DocVectorSpace object")
        titles, data = data["title"], data["text"]
        self._doc_titles = titles

        # preproces the documents, and create TaggedDocuments
        tagged_data = [
                TaggedDocument(
                    words=word_tokenize(doc.lower()),
                    tags=[str(i)]
                )
                for i, doc in enumerate(data)
        ]

        # train the Doc2vec model
        default_kwargs = dict(
                vector_size=1000,
                min_count=5,
                workers=4,
                epochs=20
        )
        kwargs = default_kwargs | kwargs

        self._model = Doc2Vec(**kwargs)
        self._model.build_vocab(tagged_data)
        self._model.train(
            tagged_data,
            total_examples=self._model.corpus_count,
            epochs=self._model.epochs
        )

        # for finding the document indexes using the doc title and vice-versa
        self.doc2index2doc = {
                k: v
                for idx, title in enumerate(data)
                for k, v in [(idx, title), (title, idx)]
        }

        # get the document vectors
        self._doc_vectors = np.asarray([
            self._model.infer_vector(word_tokenize(doc.lower()))
            for doc in data
        ])

        norms = np.linalg.norm(self._doc_vectors, axis=1)
        normalized_matrix = self._doc_vectors / norms
        self._similarity_matrix = np.dot(
                normalized_matrix, normalized_matrix.T
        )

    def get_similarity_matrix(self):
        return self._similarity_matrix

    def __getitem__(self, title: str) -> np.ndarray:
        return self._doc_vectors[self.doc2index2doc[title]]

    def __len__(self):
        return len(self._doc_titles)

    def __iterable__(self):
        self._it = 0
        return self

    def __next__(self) -> tuple[str, np.ndarray]:
        self._it += 1
        if self._it > len(self):
            raise StopIteration
        i = self._it - 1
        return self._doc_titles[i], self._doc_vectors[i]


class DocProbabilities:
    """Manages the document probabilities and sampling"""
    def __init__(self, n: int, title2idx: dict[str, int], weights: Iterable[float] | None = None):
        """Initilizes the object

        Parameters:
            n: int
                Size of the option space.

            title2idx: dict[str, int]
                Object that returns the document index when entering its title.

            weights (optional): Iterable[float] | None
                Initial vector probabilities. Must be of size N.
                If set to None, defaults to uniform probability.
        """
        LOGGER.debug("Initializing a DocProbabilities object")
        self._tti = title2idx
        self._probs = np.ones(n)/n if weights is None else weights

    def __iter__(self):
        return iter(np.copy(self._probs))

    def __getitem__(self, doc_title: str) -> float:
        return self._probs[self._tti[doc_title]]

    def __setitem__(self, doc_title: str, new_prob: float):
        self._probs[self._tti[doc_title]] = new_prob


class DocVectorManager:
    def __init__(
        self,
        doc_vecspace: DocVectorSpace,
        answer_model: BaseAnswerModel,
        doc_probabilities: DocProbabilities
    ):
        LOGGER.debug("Initializing a DocVectorManager object")
        self._d2i2d = doc_vecspace.doc2index2doc
        self._vecspace = doc_vecspace
        self._doc_prob = doc_probabilities
        self._answrmod = answer_model
        self._most_likely_guess = ("", 0)

    def get_most_likely_guess(self):
        return self._most_likely_guess

    def get_highest_entropy_doc(self):
        LOGGER.debug("Inside DocVectorManager.get_highrst_entropy_doc")
        if hasattr(BaseAnswerModel, "get_highest_entropy_doc"):
            return self._answrmod.get_highest_entropy_doc()
        if hasattr(BaseAnswerModel, "get_highest_entropy_doc_idx"):
            idx, prob = self._answrmod.get_highest_entropy_doc_idx()
            return self._d2i2d[idx], prob
        raise NotImplementedError

    def update_doc_probs(self, answer: Answer):
        LOGGER.debug("Inside DocVectorManager.update_doc_probs")
        most_likely_guess = ("", 0)
        for doc, priori in zip(self._vecspace, self._doc_prob):
            doc_prob =\
                 (answer | doc) * priori / answer.prob
            if doc_prob > most_likely_guess[1]:
                most_likely_guess = (doc, doc_prob)
            self._doc_prob[doc] = doc_prob
        self._most_likely_guess = most_likely_guess

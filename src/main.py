import sys
import argparse
import logging
from datasets import load_from_disk
from qna.models import VectorSimilarityAnswerModel, Answer
from qna.doc2vec import DocVectorSpace, DocProbabilities, DocVectorManager


def load_data(data_path):
    """load the data"""
    return load_from_disk(data_path)['train']


def get_args():
    """get program arguments"""
    parser = argparse.ArgumentParser(prog="main")
    parser.add_argument("data_path")
    parser.add_argument("goal")
    return parser.parse_args()


def main():
    logger = logging.getLogger(__name__)
    args = get_args()
    data = load_data(args.data_path)

    logger.info("Program launched")    

    vecspace = DocVectorSpace(data)

    logger.info("Vectorspace done")

    probdist = DocProbabilities(len(vecspace), vecspace.doc2index2doc)
    ans_model = VectorSimilarityAnswerModel()
    doc_mangr = DocVectorManager(vecspace, ans_model, probdist)

    print("Answer all questions with either 0 (for no) or 1 (for yes).")
    guess = None
    tries = 0
    while guess != args.goal:
        if tries > 20:
            print("NOOOOOOOOO I LOST T-T")
            sys.exit()
        word, prob = doc_mangr.get_highest_entropy_doc()
        ans_status = int(input(f"Is the goal word realted to {word}? "))

        ans = Answer(ans_status, prob, word, vecspace[word])
        doc_mangr.update_doc_probs(ans)
        guess = doc_mangr.get_most_likely_guess()
        tries += 1
        print("I won >:)")


if __name__ == '__main__':
    main()

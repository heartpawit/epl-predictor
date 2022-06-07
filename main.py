from argparse import ArgumentParser
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.decomposition import TruncatedSVD
from sklearn.ensemble import GradientBoostingClassifier
import pandas as pd
import numpy as np
from nltk import spearman_correlation
from tqdm import tqdm

from utils import generate_dataset, tf_idf, calculate_similarity, create_freqdist

def check_args(args):
    assert args.encoder in ['use', 'sentbert', 'tfidf', 'unigram', 'bigram']
    assert args.predictor in ['classifier', 'cossim', 'spearman']

    if args.encoder.endswith('gram'):
        assert args.predictor == 'spearman', f"{args.encoder} is compatible with 'spearman' only"
    else:
        assert args.predictor != 'spearman', f"{args.encoder} is compatible with either 'cossim' or 'classifier'"

def main(args):
    win, draw, lose = generate_dataset('train.csv')

    print("===> Encoding the training set")
    if args.encoder == 'use':
        import tensorflow_hub as hub

        module_url = "https://tfhub.dev/google/universal-sentence-encoder/4" 
        encoder = hub.load(module_url)

        win_set = encoder(win["Comment"])
        lose_set = encoder(lose["Comment"])
        X = np.concatenate([win_set, lose_set])
    elif args.encoder == 'sentbert':
        from sentence_transformers import SentenceTransformer, util
        
        encoder = SentenceTransformer('all-MiniLM-L6-v2')
        win_set = encoder.encode(win["Comment"], show_progress_bar=True)
        lose_set = encoder.encode(lose["Comment"], show_progress_bar=True)
        X = np.concatenate([win_set, lose_set])
    elif args.encoder == 'tfidf':
        encoder, X = tf_idf(pd.concat([win, lose]))
        win_set = encoder.transform(win["Comment"])
        lose_set = encoder.transform(lose["Comment"])
    else:
        win_set = create_freqdist(win["Comment"].to_list(), mode=args.encoder)
        lose_set = create_freqdist(lose["Comment"].to_list(), mode=args.encoder)

    if args.predictor == 'classifier':
        print("===> Training the classifier")
        svd = TruncatedSVD(n_components=128)
        svd.fit(X)

        win_set = svd.transform(win_set)
        lose_set = svd.transform(lose_set)

        clf = GradientBoostingClassifier(random_state=42)
        clf.fit(np.concatenate([win_set, lose_set]), [1] * len(win_set) + [0] * len(lose_set))

    print("===> Initializing the test set")
    test_set = generate_dataset('test.csv', mode=args.eval_mode)
    prediction = []
    ground_truth = []

    if args.eval_mode == 'single':
        for post in tqdm(test_set):
            result = (post["Result"].to_list())[0]

            if result == 0:
                continue
            elif result < 0:
                ground_truth.append(0)
            elif result > 0:
                ground_truth.append(1)

            if args.encoder == 'use':
                vectorized_test = encoder(post["Comment"])
            elif args.encoder == 'sentbert':
                vectorized_test = encoder.encode(post["Comment"])
            elif args.encoder == 'tfidf':
                vectorized_test = encoder.transform(post["Comment"])
            else:
                vectorized_test = create_freqdist(post["Comment"].to_list(), mode=args.encoder)

            if args.predictor == 'classifier':
                vectorized_test = svd.transform(vectorized_test)
                prediction.append(int(np.average(clf.predict(vectorized_test)) >= 0.5))
            elif args.predictor == 'cossim':
                win_prob = np.average(calculate_similarity(vectorized_test, win_set))
                lose_prob = np.average(calculate_similarity(vectorized_test, lose_set))
                prediction.append(np.argmax([lose_prob, win_prob]))
            else:
                win_sc = spearman_correlation(win_set, vectorized_test)
                lose_sc = spearman_correlation(lose_set, vectorized_test)
                prediction.append(np.argmax([lose_sc, win_sc]))

    elif args.eval_mode == 'double':
        for post_home, post_away in tqdm(test_set):
            result = int((post_home["Result"].to_list())[0])

            if result == 0:
                continue
            elif result < 0:
                ground_truth.append(0)
            elif result > 0:
                ground_truth.append(1)

            if args.encoder == 'use':
                vectorized_home = encoder(post_home["Comment"])
                vectorized_away = encoder(post_away["Comment"])
            elif args.encoder == 'sentbert':
                vectorized_home = encoder.encode(post_home["Comment"])
                vectorized_away = encoder.encode(post_away["Comment"])
            elif args.encoder == 'tfidf':
                vectorized_home = encoder.transform(post_home["Comment"])
                vectorized_away = encoder.transform(post_away["Comment"])
            else:
                vectorized_home = create_freqdist(post_home["Comment"].to_list(), mode=args.encoder)
                vectorized_away = create_freqdist(post_away["Comment"].to_list(), mode=args.encoder)

            if args.predictor == 'classifier':
                vectorized_home = svd.transform(vectorized_home)
                vectorized_away = svd.transform(vectorized_away)

                home_win_prob = np.average(clf.predict(vectorized_home))
                home_lose_prob = 1 - home_win_prob
                away_win_prob = np.average(clf.predict(vectorized_away))
                away_lose_prob = 1 - home_lose_prob
            elif args.predictor == 'cossim':
                home_win_prob = np.average(calculate_similarity(vectorized_home, win_set))
                home_lose_prob = np.average(calculate_similarity(vectorized_home, lose_set))
                away_win_prob = np.average(calculate_similarity(vectorized_away, win_set))
                away_lose_prob = np.average(calculate_similarity(vectorized_away, lose_set))
            else:
                vectorized_home = create_freqdist(post_home["Comment"].to_list(), mode=args.encoder)
                vectorized_away = create_freqdist(post_away["Comment"].to_list(), mode=args.encoder)
                
                home_win_prob = spearman_correlation(win_set, vectorized_home)
                home_lose_prob = spearman_correlation(lose_set, vectorized_home)
                away_win_prob = spearman_correlation(win_set, vectorized_away)
                away_lose_prob = spearman_correlation(lose_set, vectorized_away)

            home_prob = [home_lose_prob, home_win_prob]
            away_prob = [away_lose_prob, away_win_prob]

            prediction.append(np.argmax([away_prob[1] + home_prob[0], home_prob[1] + away_prob[0]]))

    print(f"Accuracy: {accuracy_score(ground_truth, prediction):.2%}")
    print(confusion_matrix(ground_truth, prediction))

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument('--eval_mode', type=str, default='single')
    parser.add_argument('--encoder', type=str, default='tfidf')
    parser.add_argument('--predictor', type=str, default='cossim')

    args = parser.parse_args()
    check_args(args)
    main(args)
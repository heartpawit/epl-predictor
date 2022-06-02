from argparse import ArgumentParser
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.decomposition import TruncatedSVD
from sklearn.ensemble import GradientBoostingClassifier
import pandas as pd
import numpy as np
from tqdm import tqdm

from utils import generate_dataset, tf_idf, calculate_similarity

mode = 'double_eval'
use_svd = True

def main(args):
    win, draw, lose = generate_dataset('train.csv')
    print("===> Performing TF-IDF on the training set")
    vectorizer, X = tf_idf(pd.concat([win, lose]))

    if use_svd:
        svd = TruncatedSVD(n_components=128)
        svd.fit(X)
        pass

    win_set = vectorizer.transform(win["Comment"])
    # draw_set = vectorizer.transform(draw["Comment"])
    lose_set = vectorizer.transform(lose["Comment"])

    if use_svd:
        print("===> Training the classifier")
        win_set = svd.transform(win_set)
        lose_set = svd.transform(lose_set)

        clf = GradientBoostingClassifier(random_state=42)
        clf.fit(np.concatenate([win_set, lose_set]), [1] * len(win_set) + [0] * len(lose_set))
        print(accuracy_score([1] * len(win_set) + [0] * len(lose_set), clf.predict(np.concatenate([win_set, lose_set]))))

    print("===> Predicting the test set")
    test_set = generate_dataset('test.csv', mode=mode)
    prediction = []
    ground_truth = []

    if mode == 'single_eval':
        for post in tqdm(test_set):
            result = (post["Result"].to_list())[0]
            if result == 0:
                continue

            elif result < 0:
                ground_truth.append(0)

            elif result > 0:
                ground_truth.append(1)

            vectorized_test = vectorizer.transform(post["Comment"])

            if use_svd:
                vectorized_test = svd.transform(vectorized_test)
                prediction.append(int(np.average(clf.predict(vectorized_test)) >= 0.5))

            else:
                win_prob = np.average(calculate_similarity(vectorized_test, win_set))
                lose_prob = np.average(calculate_similarity(vectorized_test, lose_set))

                prediction.append(np.argmax([lose_prob, win_prob]))

    elif mode == 'double_eval':
        for post_home, post_away in tqdm(test_set):
            result = (post_home["Result"].to_list())[0]
            if result == 0:
                continue

            elif result < 0:
                ground_truth.append(0)

            elif result > 0:
                ground_truth.append(1)

            vectorized_home = vectorizer.transform(post_home["Comment"])
            vectorized_away = vectorizer.transform(post_away["Comment"])

            if use_svd:
                vectorized_home = svd.transform(vectorized_home)
                vectorized_away = svd.transform(vectorized_away)

                home_win_prob = np.average(clf.predict(vectorized_home))
                home_lose_prob = 1 - home_win_prob
                away_win_prob = np.average(clf.predict(vectorized_away))
                away_lose_prob = 1 - home_lose_prob

            else:
                home_win_prob = np.average(calculate_similarity(vectorized_home, win_set))
                home_lose_prob = np.average(calculate_similarity(vectorized_home, lose_set))
                away_win_prob = np.average(calculate_similarity(vectorized_away, win_set))
                away_lose_prob = np.average(calculate_similarity(vectorized_away, lose_set))

            home_prob = [home_lose_prob, home_win_prob]
            away_prob = [away_lose_prob, away_win_prob]

            prediction.append(np.argmax([away_prob[1] + home_prob[0], home_prob[1] + away_prob[0]]))

    print(f"Accuracy: {accuracy_score(ground_truth, prediction):.2%}")
    print(confusion_matrix(ground_truth, prediction))


if __name__ == "__main__":
    parser = ArgumentParser()
    # parser.add_argument('team', type=str)

    main(parser.parse_args())

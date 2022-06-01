from argparse import ArgumentParser

from dataset import generate_dataset
from utils import tf_idf

def main(args):
    win, draw, lose = generate_dataset('train.csv')
    X = tf_idf(win + draw + lose)
    print(X[:10])
    print(X.shape)


if __name__ == "__main__":
    parser = ArgumentParser()
    # parser.add_argument('team', type=str)

    main(parser.parse_args())
from nltk import word_tokenize, FreqDist, spearman_correlation, bigrams, trigrams
from nltk.corpus import stopwords
from utils import generate_dataset
from sklearn.metrics import confusion_matrix

import numpy as np
import re
from tqdm import tqdm

stopws = stopwords.words('english')

def create_freqdist(sents, mode='unigram'):
    tokens = [token.lower() for comment in sents
              for token in word_tokenize(comment)
              if token not in stopws
              and not re.fullmatch('\W+', token)]
    if mode == 'unigram':    
        return FreqDist(tokens)
    elif mode == 'bigram':
        bg = bigrams(tokens)
        return FreqDist(bg)

def main():
    freqdist_mode = 'unigram'
    eval_mode = 'double_eval'
    
    win, draw, lose = generate_dataset('train.csv')
    win_fd = create_freqdist(win['Comment'].to_list(), mode=freqdist_mode)
    # draw_fd = create_freqdist(draw['Comment'].to_list())
    lose_fd = create_freqdist(lose['Comment'].to_list(), mode=freqdist_mode)
    
    res = []
    test = generate_dataset('test.csv', mode=eval_mode)
    for post in tqdm(test):
        if eval_mode == 'single_eval':
            result = int(post['Result'].to_list()[0])
            if result == 0:
                continue
            
            fd = create_freqdist(post['Comment'].to_list(), mode=freqdist_mode)
            win_sc = spearman_correlation(win_fd, fd)
            lose_sc = spearman_correlation(lose_fd, fd)
            
            pred = np.argmax([lose_sc, win_sc]) * 2 - 1
            res.append((pred, -1 if result < 0 else 1))
            
        elif eval_mode == 'double_eval':
            home_post, away_post = post
            result = int(home_post['Result'].to_list()[0])
            if result == 0:
                continue
            
            home_fd = create_freqdist(home_post['Comment'].to_list(), mode=freqdist_mode)
            away_fd = create_freqdist(away_post['Comment'].to_list(), mode=freqdist_mode)
            
            home_win_sc = spearman_correlation(win_fd, home_fd)
            home_lose_sc = spearman_correlation(lose_fd, home_fd)
            away_win_sc = spearman_correlation(win_fd, away_fd)
            away_lose_sc = spearman_correlation(lose_fd, away_fd)
            
            home_sc = np.array([home_lose_sc, home_win_sc]) # / (home_lose_sc + home_win_sc)
            away_sc = np.array([away_lose_sc, away_win_sc]) # / (away_lose_sc + away_win_sc)
            
            pred = np.argmax([home_sc[0] + away_sc[1], home_sc[1] + away_sc[0]]) * 2 - 1
            # print(pred, result)
            res.append((pred, -1 if result < 0 else 1))
    
    # print(res)
    acc = len(list(filter(lambda x : x[0] == x[1], res)))/ len(res)
    print("Accuracy: {:.2f}%".format(acc*100))
    print(confusion_matrix([x[1] for x in res], [x[0] for x in res]))
    
if __name__ == '__main__':
    main()
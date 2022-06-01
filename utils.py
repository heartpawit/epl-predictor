import pandas as pd
from datetime import datetime
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk import word_tokenize

def get_match_times(team):
    fixtures = pd.read_csv('epl_fixtures.csv')
    team_fixtures = fixtures.loc[(fixtures['Home Team'] == team) | (fixtures['Away Team'] == team)]
    match_times = [datetime.strptime(x, '%d/%m/%Y %H:%M') for x in team_fixtures['Date']]
    return match_times

def is_valid_post(post, match_times):
    post_time = post["time"]

    for time in match_times:
        if abs((post_time - time).total_seconds()) <= 86400:
            return True
    return False

def tf_idf(comments):
    comments = [comment['comment'] for comment in comments]
    vectorizer = TfidfVectorizer(tokenizer=word_tokenize, stop_words='english', lowercase=False)
    X = vectorizer.fit_transform(comments)
    return vectorizer, X

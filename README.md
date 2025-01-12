# epl-predictor

> Who Wins? Utilizing Facebook Comments for Football Match Predictions
> 
> CS372 Natural Language Processing with NLP, KAIST (Spring 2022)

Check our [paper](who_wins.pdf).

## Introduction

We propose a novel approach to predict football match results which depends only on Facebook public comments posted *before* the match starts 
and does not require previous statistics that may include expensive access or require intensive feature engineering.

## Dataset

**EPC20-21 Dataset**: Facebook comments in the starting line-up updates from 405 posts across 16 teams and 301 matches 
in the English Premier League 2020-21 season.

| Labels  | Number of Comments |
| ------- | ------------------ |
| Win     | 19392              |
| Draw    | 13352              |
| Lose    | 20388              |

## Dependencies

Install the following libraries:
- nltk
- pandas
- scikit-learn
- tqdm
- [facebook_scraper](https://github.com/kevinzg/facebook-scraper)

Optional libraries (using semantic-based encoder):
- tensorflow-hub
- sentence-transformers

## Usage

### Running the baselines

Example:
```
python main.py --encoder unigram --predictor spearman --eval_mode single
python main.py --encoder tfidf --predictor cossim --eval_mode single
python main.py --encoder sentbert --predictor cossim --eval_mode double
```

### Scraping your own dataset

Run the following commands:
```
# Loop your preferred teams to be included in the dataset
python scraper/page_scraper.py [team]

# Manually arrange post IDs as shown in the epl_fixtures-post_ids.csv file
python scraper/post_scraper.py
```

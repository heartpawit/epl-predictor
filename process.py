import csv
import os
import json

def main():
    data_dir = 'posts'
    
    with open('train.csv', mode='w', newline='', encoding='utf-8') as train_csv:
        train_writer = csv.writer(train_csv)
        train_writer.writerow(['Comment', 'Match_ID', 'Home_Team', 'Away_Team', 'Comment_Team', 'Result'])
        
        with open('epl_fixtures-post_ids.csv', mode='r') as fixture_csv:
            fixture_reader = csv.DictReader(fixture_csv)
            for match in fixture_reader:
                result = match['Results'].split(' - ')
                if match['Home_PostID']:
                    file_dir = os.path.join(data_dir, match['Home_PostID'] + '.json')
                    with open(file_dir) as fp:
                        comments = json.load(fp)['comments_full']
                        for comment in comments:
                            train_writer.writerow([
                                comment['comment_text'],
                                match['Match Number'],
                                match['Home'],
                                match['Away'],
                                'Home',
                                int(result[0]) - int(result[1])
                            ])
                if match['Away_PostID']:
                    file_dir = os.path.join(data_dir, match['Away_PostID'] + '.json')
                    with open(file_dir) as fp:
                        comments = json.load(fp)['comments_full']
                        for comment in comments:
                            train_writer.writerow([
                                comment['comment_text'],
                                match['Match Number'],
                                match['Home'],
                                match['Away'],
                                'Away',
                                int(result[0]) - int(result[1])
                            ])
                    
        
if __name__ == '__main__':
    main()
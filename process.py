import csv
import os
import json

test_match_ids = [71, 144, 367, 267, 281, 311, 279, 88, 352, 157,
                  133, 34, 257, 238, 147, 368, 124, 148, 346, 117,
                  114, 362, 119, 250, 301, 152, 306, 205, 132, 100, 
                  22, 197, 61, 121, 38, 59, 324, 120, 372, 318]

def main():
    data_dir = 'posts'
    
    with open('train.csv', mode='w', newline='', encoding='utf-8') as train_csv, open('test.csv', mode='w', newline='', encoding='utf-8') as test_csv:
        train_writer = csv.writer(train_csv)
        train_writer.writerow(['Comment', 'Match_ID', 'Home_Team', 'Away_Team', 'Result'])
        test_writer = csv.writer(test_csv)
        test_writer.writerow(['Comment', 'Match_ID', 'Home_Team', 'Away_Team', 'Result'])
        
        with open('epl_fixtures-post_ids.csv', mode='r') as fixture_csv:
            fixture_reader = csv.DictReader(fixture_csv)
            for match in fixture_reader:
                result = match['Results'].split(' - ')                    
                if match['Home_PostID']:
                    file_dir = os.path.join(data_dir, match['Home_PostID'] + '.json')
                    with open(file_dir) as fp:
                        comments = json.load(fp)['comments_full']
                        for comment in comments:
                            row = [comment['comment_text'],
                                   match['Match Number'],
                                   match['Home'],
                                   match['Away'],
                                #    'Home',
                                   int(result[0]) - int(result[1])]
                            if int(match['Match Number']) in test_match_ids:
                                test_writer.writerow(row)
                            else:
                                train_writer.writerow(row)
                                
                if match['Away_PostID']:
                    file_dir = os.path.join(data_dir, match['Away_PostID'] + '.json')
                    with open(file_dir) as fp:
                        comments = json.load(fp)['comments_full']
                        for comment in comments:
                            row = [comment['comment_text'],
                                   match['Match Number'],
                                   match['Home'],
                                   match['Away'],
                                #    'Away',
                                   int(result[1]) - int(result[0])]
                            if match['Match Number'] in test_match_ids:
                                test_writer.writerow(row)
                            else:
                                train_writer.writerow(row)
                    
        
if __name__ == '__main__':
    main()
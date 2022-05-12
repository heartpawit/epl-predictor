from facebook_scraper import get_posts
import csv
import os
import json

options = {
    # "progress": True,
    "comments": True,
    "allow_extra_requests": False,
    "reactors": True,
    # "posts_per_page": 200,
}

def main():
    save_dir = 'posts'

    with open('epl_fixtures-post_ids.csv', mode='r') as csvfile:
        csvreader = csv.DictReader(csvfile)
        matches = list(csvreader)
        home_postIDs = [match['Home_PostID'] for match in matches if match['Home_PostID']]
        away_postIDs = [match['Away_PostID'] for match in matches if match['Away_PostID']]
        postIDs = home_postIDs + away_postIDs

        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        
        for post in get_posts(post_urls=postIDs, cookies='facebook.com_cookies.txt', timeout=300, options=options):
            file_dir = os.path.join(save_dir, post["original_request_url"] + '.json')
            with open(file_dir, 'w') as fp:
                json.dump(post, fp, default=str, indent=4)
                print(f'{post["post_id"]}.json saved')
    
    
if __name__ == "__main__":
    main()
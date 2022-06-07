from facebook_scraper import get_posts
import csv
import os
import json

options = {
    # "progress": True,
    "comments": 150,
    "allow_extra_requests": False,
    # "reactors": True,
    # "posts_per_page": 200,
}

def main():
    save_dir = 'posts'

    with open('epl_fixtures-post_ids.csv', mode='r') as csvfile:
        csvreader = csv.DictReader(csvfile)
        matches = list(csvreader)
        home_postIDs = [match['Home_PostID'] for match in matches if match['Home_PostID']]
        away_postIDs = [match['Away_PostID'] for match in matches if match['Away_PostID']]
        postIDs = set(home_postIDs + away_postIDs)

        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        scraped_files = set([os.path.splitext(path)[0] for path in os.listdir(save_dir)])
        postIDs -= scraped_files
        print(f"Attempting to scrape {len(postIDs)} files")
        
        for post in get_posts(post_urls=postIDs, timeout=300, options=options):
        # for post in get_posts(post_urls=postIDs, cookies='facebook.com_cookies.txt', timeout=300, options=options):
            file_dir = os.path.join(save_dir, post["original_request_url"] + '.json')
            if 'text' in post.keys():
                with open(file_dir, 'w') as fp:
                    json.dump(post, fp, default=str, indent=4)
                    print(f'{post["original_request_url"]}.json saved')
            else:
                print(f"Skipping {post['original_request_url']}")

    
if __name__ == "__main__":
    main()
from facebook_scraper import get_posts
from argparse import ArgumentParser
import os
import json

from utils import get_match_times, is_valid_post

options = {
    # "progress": True,
    "comments": True,
    "allow_extra_requests": True,
    "reactors": True
}

def main(args):
    team = args.team
    team_dir = os.path.join('data', team.lower().replace(' ', '_'))

    print('===> Loading team fixtures')
    match_times = get_match_times(team)
    min_time = min(match_times)
    max_time = max(match_times)
    
    print('===> Loading page name')
    try:
        fp = open('page_names.json')
        page = json.load(fp)[team]
        fp.close()
    except:
        raise FileNotFoundError    

    if not os.path.exists(team_dir):
        os.makedirs(team_dir)

    print('===> Start scraping')
    for post in get_posts(page, cookies='facebook.com_cookies.txt', page=None, options=options):
        if post['time'] < min_time:
            print('Search completed')
            break

        if post['time'] > max_time:
            continue

        if is_valid_post(post, match_times):
            file_dir = os.path.join(team_dir, post["post_id"] + '.json')
            with open(file_dir, 'w') as fp:
                json.dump(post, fp, default=str, indent=4)
                print(f'{post["post_id"]}.json saved to {team_dir}')

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument('team', type=str)

    main(parser.parse_args())
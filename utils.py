import pandas as pd
from datetime import datetime

def get_match_times(team):
    fixtures = pd.read_csv('epl_fixtures.csv')
    team_fixtures = fixtures.loc[(fixtures['Home Team'] == team) | (fixtures['Away Team'] == team)]
    return list(team_fixtures['Date'].apply(lambda x: datetime.strptime(x, '%d/%m/%Y %H:%M')))

def is_valid_post(post, match_times):
    post_time = datetime.strptime(post["time"], '%d/%m/%Y %H:%M')

    for time in match_times:
        if abs((post_time - time).seconds()) <= 1440:
            return True
    return False

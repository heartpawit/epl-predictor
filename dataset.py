import csv


def generate_dataset(file):
    with open(file, mode='r', newline='', encoding='utf-8') as f:
        csvreader = csv.DictReader(f)
        comments = list(csvreader)

    win = [{'comment': comment['Comment'], 'goal_diff': int(
        comment['Result'])} for comment in comments if int(comment['Result']) > 0]
    draw = [{'comment': comment['Comment'], 'goal_diff': int(comment['Result'])}
            for comment in comments if int(comment['Result']) == 0]
    lose = [{'comment': comment['Comment'], 'goal_diff': int(comment['Result'])}
            for comment in comments if int(comment['Result']) < 0]

    return win, draw, lose


if __name__ == '__main__':
    w, d, l = generate_dataset('train.csv')
    print(w[:5], '\n', d[:5], '\n', l[:5])

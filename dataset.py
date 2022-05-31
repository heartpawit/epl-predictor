import csv

def generate_dataset(file):
    with open(file, mode='r', newline='', encoding='utf-8') as f:
        csvreader = csv.DictReader(f)
        comments = list(csvreader)

    win = [(comment['Comment'], int(comment['Result'])) for comment in comments if int(comment['Result']) > 0]
    draw = [(comment['Comment'], int(comment['Result'])) for comment in comments if int(comment['Result']) == 0]
    lose = [(comment['Comment'], int(comment['Result'])) for comment in comments if int(comment['Result']) < 0]
    
    return win, draw, lose

if __name__ == '__main__':
    w, d, l = generate_dataset('train.csv')
    print(w[:5], '\n', d[:5], '\n', l[:5])
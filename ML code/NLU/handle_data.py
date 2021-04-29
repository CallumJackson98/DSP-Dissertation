import csv
import random
import json

new_tweets = []

train_tweets_list = []     #train with 2/3     5 or 10 fold???
test_tweets = []      #test with 1/6
tune_tweets = []      #tune with 1/6



with open("E:\\Uni Work\\Uni\\Year 3\\Digital Systems Project\\Project\\Dataset\\training.1600000.processed.noemoticon.csv") as csvfile:
    reader = csv.reader(csvfile, delimiter = ',')
    count = 0
    first = next(reader)
    for row in reader:
        tweet = row[5]
        if '@' in tweet:
            # remove @s - replies
            tweet = ''.join([ i + ' ' for i in tweet.split(' ') if "@" not in i ]).strip()
        new_tweets.append([row[0], tweet])


random.shuffle(new_tweets)
count = 0
for n in new_tweets:
    if count < 1066666:
        train_tweets_list.append(n)
    elif 106666 <= count <= 1333332:
        test_tweets.append(n)
    else:
        tune_tweets.append(n)
    count+=1

pos = []
neg = []

pos_small = []
neg_small = []


for t in train_tweets_list:
    if t[0] == '0':
        neg.append(t[1])
        if len(neg_small) < 10000:
            neg_small.append(t[1])
    
    elif t[0] == '4':
        pos.append(t[1])
        if len(pos_small) < 10000:
            pos_small.append(t[1])




train_tweets = {
    "intents": [
        {
        "tag": "positive",
        "patterns": pos,
        "response": "4"
        },
        {
        "tag": "negative",
        "patterns": neg,
        "response": "0"
        }
    ]
}

train_tweets_small = {
    "intents": [
        {
        "tag": "positive",
        "patterns": pos_small,
        "response": "4"
        },
        {
        "tag": "negative",
        "patterns": neg_small,
        "response": "0"
        }
    ]
}


with open('E:\\Uni Work\\Uni\\Year 3\\Digital Systems Project\\Project\\Dataset\\new_dataset\\train_tweets.json', 'w', encoding='utf-8') as f:
    json.dump(train_tweets, f, ensure_ascii=False, indent=4)


with open('E:\\Uni Work\\Uni\\Year 3\\Digital Systems Project\\Project\\Dataset\\new_dataset\\train_tweets_small.json', 'w', encoding='utf-8') as f:
    json.dump(train_tweets_small, f, ensure_ascii=False, indent=4)




with open('E:\\Uni Work\\Uni\\Year 3\\Digital Systems Project\\Project\\Dataset\\new_dataset\\test_tweets.csv', 'w', newline = '', encoding='utf-8') as csvfile:
    writer = csv.writer(csvfile, delimiter=',')
    for t in test_tweets:
        writer.writerow(t)



with open('E:\\Uni Work\\Uni\\Year 3\\Digital Systems Project\\Project\\Dataset\\new_dataset\\tune_tweets.csv', 'w', newline = '', encoding='utf-8') as csvfile:
    writer = csv.writer(csvfile, delimiter=',')
    for t in tune_tweets:
        writer.writerow(t)
    


    
    

    

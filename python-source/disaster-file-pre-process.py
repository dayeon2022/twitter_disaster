import snscrape.modules.twitter as sntwitter
from datetime import datetime, timedelta
import pandas as pd

df = pd.read_excel("2020년 지자체 사고 8종 마이크로데이터-수난.xlsx", sheet_name="수난")
df1 = df.iloc[:,[1,2,3,4,5,16,17]]

date_format = '%Y-%m-%d'
target_list = []
check_row = 0
search_date = ''
tweet_read = []
for index, disaster in df1.iterrows():
    from_time = int(disaster[3][0:2])
    to_time = int(disaster[3][3:5])
    since_date = datetime.strptime(disaster[0],date_format)
    until_date = since_date + timedelta(days=1)
    
    print(since_date + '~' + until_date)
    
    if check_row % 10 == 0 :
        print('current row = ' + str(check_row))
    
    query = '태풍 since:'+since_date.strftime('%Y-%m-%d')+' until:' + until_date.strftime('%Y-%m-%d')
    
    if search_date != since_date :
        search_date = since_date
        tweet_read = []
        for tweet in sntwitter.TwitterSearchScraper(query).get_items():
            tweet_read.append(tweet)
            tweet_hour = tweet.date.hour
            if tweet_hour >= from_time and tweet_hour <= to_time :
                list = [disaster[0],disaster[1],disaster[2],disaster[3],disaster[4],disaster[5],disaster[6],disaster[7],tweet.content]
                target_list.append(list)
    else:
        for tweet in tweet_read:
            tweet_hour = tweet.date.hour
            if tweet_hour >= from_time and tweet_hour <= to_time :
                list = [disaster[0],disaster[1],disaster[2],disaster[3],disaster[4],disaster[5],disaster[6],disaster[7],tweet.content]
                target_list.append(list)
    
    check_row = check_row + 1

target_df = pd.DataFrame(target_list, columns=['신고년월일','월','신고시각','시간분류','발생장소','사망자수','부상자수','위험도','트윗'])
target_df
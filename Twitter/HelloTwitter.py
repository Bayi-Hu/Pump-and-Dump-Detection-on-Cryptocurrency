import pandas as pd
import tweepy
import csv

consumer_key = "O7n0nwPucJTHUYjz3GGaYQJhB"
consumer_secret = "ADTXTADfcBFARYP2Digmq9YvLhnZyMLefawYbOrG0LP0fxo51n"
access_token = "1362613168992845830-sbJFtV9jlVOOop33dsh8OYuPQ3XqCB"
access_token_secret = "laJm5bGfLVXAkkbTHcdfcD0VFegvKhn68YEqBn62ldEqc"

# auth = tweepy.OAuthHandler(consumer_key, consumer_secret)
# auth.set_access_token(access_token, access_token_secret)
# api = tweepy.API(auth)
auth = tweepy.AppAuthHandler(consumer_key, consumer_secret)
api = tweepy.API(auth)

public_tweets = api.home_timeline()

for tweet in public_tweets:
    print(tweet)



# tweet_url = pd.read_csv("Your_Text_File.txt", index_col= None,
# header = None, names = ["links"])
#
# af = lambda x: x["links"].split("/")[-1]
# tweet_url['id'] = tweet_url.apply(af, axis=1)
# tweet_url.head()
#
# ids = tweet_url['id'].tolist()
# total_count = len(ids)
# chunks = (total_count - 1) // 50 + 1
#
# def fetch_tw(ids):
#     list_of_tw_status = api.statuses_lookup(ids, tweet_mode= "extended")
#     empty_data = pd.DataFrame()
#     for status in list_of_tw_status:
#             tweet_elem = {"date": status.created_at,
#                      "tweet_id":status.id,
#                      "tweet":status.full_text,
#                      "User location":status.user.location,
#                      "Retweet count":status.retweet_count,
#                      "Like count":status.favorite_count,
#                      "Source":status.source}
#             empty_data = empty_data.append(tweet_elem, ignore_index = True)
#     empty_data.to_csv("new_tweets.csv", mode="a")
#
# for i in range(chunks):
#         batch = ids[i*50:(i+1)*50]
#         result = fetch_tw(batch)

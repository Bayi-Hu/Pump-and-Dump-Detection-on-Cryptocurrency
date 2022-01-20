import tweepy

bearer_token = "AAAAAAAAAAAAAAAAAAAAAFoNXgEAAAAA0XJPBtcF5XveM8inL5HR7EHucMo%3DdN7kRzY1ugUyJy26pkPL4rGfssJnaSB4MEsGsZVdf9YiIqQi5Z"
client = tweepy.Client(bearer_token=bearer_token)

# Replace with your own search query
# query = 'from:suhemparack -is:retweet'

query = '$LEV -is:retweet lang:en'

next_token = None
for i in range(10):

    tweets = client.search_recent_tweets(query=query, tweet_fields=['context_annotations', 'created_at'], max_results=10, next_token= next_token)
    for tweet in tweets.data:
        print(tweet.text)
        if len(tweet.context_annotations) > 0:
            print(tweet.context_annotations)

    next_token = tweets.meta["next_token"]
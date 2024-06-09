
# Enter API tokens below
bearer_token = 'AAAAAAAAAAAAAAAAAAAAAMbGsgEAAAAAn4QTKa4AYys7thVh7mBR9%2BfrJ1E%3DVsgdBECkFyUbkINnQNj27uDbPNB6oftqmo7HGMq7SGCStP83KJ'
consumer_key = 'Kab2pPS5CWfAtWGHbLPq69W7a'
consumer_secret = 'W6Kg2xCE2hyz95QlI1PpBBTuA63UeZJOj6xlVea7llqwBNKGvb'
access_token = '1765806016120164353-1T4mgfIUNpLGQM6TV5EEZ9Hxk5q19l'
access_token_secret = 'jDDtvOsOwdB2uXHctnqUF31ZHtUaLvlxmdjdjDyG79ZEp'
import tweepy

# V1 Twitter API Authentication
auth = tweepy.OAuthHandler(consumer_key, consumer_secret)
auth.set_access_token(access_token, access_token_secret)
api = tweepy.API(auth, wait_on_rate_limit=True)

# V2 Twitter API Authentication
client = tweepy.Client(
    bearer_token,
    consumer_key,
    consumer_secret,
    access_token,
    access_token_secret,
    wait_on_rate_limit=True,
)

def PostMarsaHamrunTweet(Traffic0, Traffic1, Traffic2):
    # Upload image to Twitter. Replace 'filename' your image filename.
    media_id = api.media_upload(filename="MarsaHamrun.jpg").media_id_string
    text = f"Direction Marsa: {Traffic0}, Direction San Gwann: {Traffic1}, Direction Qormi: {Traffic2}"
    client.create_tweet(text=text, media_ids=[media_id])
    print("Tweeted!")

def PostMsidaTweet(Traffic0, Traffic1, Traffic2, Traffic3):
    # Upload image to Twitter. Replace 'filename' your image filename.
    media_id = api.media_upload(filename="Msida.jpg").media_id_string
    text = f"Direction Mater Dei: {Traffic0}, Direction Santa Venera: {Traffic1}, Direction Hamrun: {Traffic2}, Direction Santa Venera {Traffic3}"
    client.create_tweet(text=text, media_ids=[media_id])
    print("Tweeted!")

def PostQormiTweet(Traffic0):
    media_id = api.media_upload(filename="QormiMdinaRoad.jpg").media_id_string
    text = f"Mdina Road Roundabout: {Traffic0}"
    client.create_tweet(text=text, media_ids=[media_id])
    print("Tweeted!")

def PostQormiCTweet(Traffic0):
    media_id = api.media_upload(filename="QormiCanonRoad.jpg").media_id_string
    text = f"Direction Attard: {Traffic0}, Direction Hamrun: {Traffic1}"
    client.create_tweet(text=text, media_ids=[media_id])
    print("Tweeted!")
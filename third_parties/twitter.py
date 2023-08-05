import os

from datetime import datetime, timezone
import tweepy


def scrape_user_tweets(username: str, num_tweets: int = 10) -> list:
    """
    Scrapes Twitter user's original tweets (i.e., no retweets or replies) and returns them as a list of dictionaries.
    Each dictionary has three fields: "time_posted" (relative or now), "text", and "url".
    :param username: Twitter account username
    :param num_tweets: number of tweets to scrape
    :return: list
    """
    auth = tweepy.OAuthHandler(
        os.environ["TWITTER_API_KEY"], os.environ["TWITTER_API_SECRET"]
    )
    auth.set_access_token(
        os.environ["TWITTER_ACCESS_TOKEN"], os.environ["TWITTER_ACCESS_SECRET"]
    )
    twitter_api = tweepy.API(auth)
    tweets = twitter_api.user_timeline(screen_name=username, count=num_tweets)
    tweet_list = []
    for tweet in tweets:
        if "RT @" not in tweet.text and not tweet.text.startswith("@"):
            tweet_dict = dict()
            tweet_dict["time_posted"] = str(
                datetime.now(timezone.utc) - tweet.created_at
            )
            tweet_dict["text"] = tweet.text
            tweet_dict[
                "url"
            ] = f"https://twitter.com/{tweet.user.screen_name}/status/{tweet.id}"

            tweet_list.append(tweet_dict)
    return tweet_list

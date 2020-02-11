# -*- coding: utf-8 -*-

import tweepy as tw
import pandas as pd
import re
import credentials as cfg
import nltk
import itertools
import collections

#Authenticate
auth = tw.OAuthHandler(cfg.consumer_key, cfg.consumer_secret)
auth.set_access_token(cfg.access_token, cfg.access_token_secret)
api = tw.API(auth, wait_on_rate_limit=True)



# Get tweets and turn them into a dataframe
def getTweets(search_words,st_date,items):
    tweets =  tw.Cursor(api.search,
              q=search_words + " -filter:retweets",
              lang="en",
              since=st_date).items(items)
     
    twits = [[tweet.user.screen_name, tweet.user.location,tweet.text,tweet.coordinates] for tweet in tweets]
    return pd.DataFrame(data=twits, columns=['user', "location","text","coordinates"])


tweets = getTweets("#RIPMoi","2020-04-10",2)


#tweets = pd.read_csv("ripMoi.csv")




#CLEAN  TEXT DATA BY REMOVING SYMBOLS AND HASHTAGS AND URL LINKS


# 1. Remove url links
def remove_links(txt):
    url_reg = r'\w+:\/{2}[\d\w-]+(\.[\d\w-]+)*(?:(?:\/[^\s/]*))*'
    return re.sub(url_reg, '', txt)

# 2. Remove words with symbols and hashtags
# '(@\S+) | (#\S+) | (\^\S+) ' Match @  or #  0r ^ and match as many non space characters and a single space.
def remove_symbols(txt):
    sym_reg = r'(@\S+)|(#\S+)|(\^\S+)'
    return re.sub(sym_reg, r'', txt)

# 3. Remove symbols  that may be in the text owing to the fact that symbols are bad for NLP
def remove_all_symbols(txt):
    reg = '!@#$%&*?[]()=+-~:;".,'
    return ''.join( c for c in txt if  c not in reg )

#4. Remove Emojis
def remove_emojis(txt):
    return txt.encode('ascii', 'ignore').decode('ascii')
    



tweet_url   = [remove_links(tweet) for tweet in tweets["text"]]
tweet_sym   = [remove_symbols(tweet) for tweet in tweet_url]
tweet_clean = [remove_all_symbols(tweet) for tweet in tweet_sym]
tweet_clean = [remove_emojis(tweet) for tweet in tweet_clean]


# Make all elements in the list lowercase
tweet_clean = [word.lower() for word in tweet_clean]



from nltk.corpus import stopwords
#nltk.download('stopwords')
stop_words = set(stopwords.words('english'))



# View a few words from the set
list(stop_words)[0:5]
#Notice that all stopwords are in lower-case



#Split words in each string
split_words = [tweet.lower().split() for tweet in tweet_clean]

#Remove all stopwords
def removeStopWords():
    tweets = [[word for word in tweet_words if not word in stop_words]
              for tweet_words in split_words]
    return tweets

tweet_clean_split = removeStopWords()


#  Now we need to combine all words in the lists and have unique values only, the faster way is to use itertools
# to flatten our lists

# A list of all words used in the tweets we've obtained
all_words = list(itertools.chain(*tweet_clean_split))

# Create counter
count_all_words = collections.Counter(all_words)

#check first 15
count_all_words.most_common(15)


# Create a dataframe of the words and their counter
tweet_clean_split_dataframe = pd.DataFrame(count_all_words.most_common(),columns=['words', 'count'])
tweet_clean_split_dataframe.head()



collection_of_words = ['president', 'moi', 'arap','mzee','daniel','toroitich','95','kenya','ya','nyayo']
tweet_clean_split = [[w for w in word if not w in collection_of_words]
                 for word in tweet_clean_split]



from nltk.tokenize import TweetTokenizer
# using strip_handles and reduce_len parameters by removing special characters
tknzr = TweetTokenizer(strip_handles=True)

nltk.download('punkt')
tweet_tokens = tknzr.tokenize(tweets['text'][0])

tweets_sent = tweets.apply(lambda row: nltk.word_tokenize(row['text']), axis=1)



# SENTIMENT ANALYSIS
# pip install textblob

from textblob import TextBlob 

# Create a textblob
blobObject = [TextBlob(tweet) for tweet in tweet_clean]

#blobObject[0].polarity, blobObject[0]

# create a list polarity values
sentiment_polar_values = [[tweet.sentiment.polarity, str(tweet)] for tweet in blobObject]

# Create a dataframe of polarity at corresponding tweet
sentiment_df = pd.DataFrame(sentiment_polar_values, columns=["polarity", "tweet"])

# positive tweets
p_ve = sentiment_df[sentiment_df.polarity > 0]

# negative tweets
n_ve = sentiment_df[sentiment_df.polarity < 0]

# neutral tweets
neut = sentiment_df[sentiment_df.polarity == 0]


print("Positive tweets percentage: {} %".format(100*len(p_ve)/len(tweet_clean))) 
print("Negative tweets percentage: {} %".format(100*len(n_ve)/len(tweet_clean))) 
print("Neutral tweets percentage: {} %".format(100*len(neut)/len(tweet_clean))) 




# PLOT THE POLARITY VALUES
import matplotlib.pyplot as plt
import seaborn as sns

fig, ax = plt.subplots(figsize=(12, 6))

sentiment_df.hist(bins=[-1, -0.75, -0.5, -0.25, 0, 0.25, 0.5, 0.75, 1],ax=ax, color="blue")

plt.title("Sentiments polarity")
plt.show()



#PLOT COMMONLY USED WORDS

plot_val= pd.DataFrame(count_all_words.most_common(15),columns=['words', 'count'])
sns.barplot(x="count", y="words", data=plot_val)

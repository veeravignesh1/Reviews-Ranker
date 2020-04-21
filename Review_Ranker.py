#!/usr/bin/env python
# coding: utf-8

# ## Flow
# 
# - Create the GUI Outline Needed
#         - Fixed Size
#         - White Background
#         - Blue Bar with Review Ranker Text
#         - (Try adding image in that bar)
# - Put in all the necessary functions
#         - get_reviews -> Input URL get df , product_name as global variable
#         - create_features -> Input df get new features
#         - clean_review -> Input df with created features -> Cleans the Review_Text column (Applying NLP techniques)
#         - predictor -> Create a Tf-Idf Matrix based of Review_Text column and Use it to make prediction
#         - ranker -> Rank the review based on the predicted values
#         - Save those files into the user specified folder under submit button def
# - Check by passing a location
#         - If empty os.getcwd()
#         - ask for folder location to save--> if given create a folder in the name of pdt and save all the 3 files
# - After saving make open folder appear
# - Look for options to display progress bar (Has to be hardcoded)

# In[1]:


###################################################################################
# Module Imports

# Review Scraping Modules
import selenium
from selenium.webdriver import Chrome, ChromeOptions
from selenium import webdriver
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.common.by import By
from selenium.common.exceptions import TimeoutException, WebDriverException
from selenium.webdriver.chrome.options import Options
import pandas as pd
import numpy as np

# Create Feature Modules
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import spacy
nlp = spacy.load("en_core_web_sm")
import re
import emoji

# Predictor Modules
from sklearn.feature_extraction.text import TfidfVectorizer

#Ranker Module
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor


# General
import warnings
warnings.filterwarnings('ignore')


####################################################################################
# Function Block

# Get Reviews


def get_review(user_url):
    '''Extracts reviews from user given `flipkart` product page and returns a `pandas dataframe`.

    Parameters
    -----------
    url: Product for which user wants to extract the review
    pages: Number of Pages of reviews the user likes to extract.By default `get_review`
    extracts 25 pages or 250 reviews

    Example
    -------
    >>> df=get_review("https://www.flipkart.com/redmi-8-ruby-red-64-gb/p/itmef9ed5039fca6?pid=MOBFKPYDCVSCZBYR")'''
    global product_name
    pages = 5  # change back to 25
    # User entered url
    url = user_url
    if 'flipkart' in url:
        review_url = url.replace('/p/', '/product-reviews/')

    # Browser Options
    options = Options()
    options.add_argument("--headless")
    options.add_argument('start-maximized')

    # Driver essential to run automated chrome window
    # No option because its in currdir
    driver = webdriver.Chrome(options=options)
    Review_Title, Review_Text, Review_Rating, Upvote, Downvote, Num_Photos = [], [], [], [], [], []

    # Extracting 25 pages of review
    for i in range(1, pages+1):

        # Change web Page
        ping = f'{review_url}&page={i}'
        driver.execute_script('window.open("{}","_self");'.format(ping))

        WebDriverWait(driver, 10).until(EC.staleness_of)

        # Check Read More Buttons
        read_more_btns = driver.find_elements_by_class_name('_1EPkIx')

        # Click on all read more in the current page
        for rm in read_more_btns:
            driver.execute_script("return arguments[0].scrollIntoView();", rm)
            driver.execute_script("window.scrollBy(0, -150);")
            rm.click()

        # Get the product name to save contents inside this folder
        if i == 1:
            product_name = driver.find_element_by_xpath(
                "//div[@class='o9Xx3p _1_odLJ']").text

        # Extracting contents
        # col _390CkK _1gY8H-
        for block in driver.find_elements_by_xpath("//div[@class='col _390CkK _1gY8H-']"):
            Review_Title.append(block.find_element_by_xpath(
                ".//p[@class='_2xg6Ul']").text)
            Review_Text.append(block.find_element_by_xpath(
                ".//div[@class='qwjRop']").text)
            Review_Rating.append(block.find_element_by_xpath(
                ".//div[@class='hGSR34 E_uFuv'or @class='hGSR34 _1x2VEC E_uFuv' or @class='hGSR34 _1nLEql E_uFuv']").text)
            Upvote.append(block.find_element_by_xpath(
                ".//div[@class='_2ZibVB']").text)
            Downvote.append(block.find_element_by_xpath(
                ".//div[@class='_2ZibVB _1FP7V7']").text)
            Num_Photos.append(len(block.find_elements_by_xpath(
                ".//div[@class='_3Z21tn _2wWSCV']")))

    # Creating df of reviews
    df = pd.DataFrame(data=list(zip(Review_Title, Review_Text, Review_Rating, Upvote, Downvote, Num_Photos)), columns=[
                      'Review_Title', 'Review_Text', 'Review_Rating', 'Upvote', 'Downvote', 'Num_Photos'])

    # Handling dtypes of Review_Rating,Upvote,Downvote
    for i in ['Review_Rating', 'Upvote', 'Downvote','Num_Photos']:
        df[i] = df[i].astype("int")
    # Return dataframe
    return df

#==================================================================================##
# Create Features

# *******Sub Funtions********

# 1. Sentiment

def sentimental_score(sentence):
    analyzer = SentimentIntensityAnalyzer()
    vs = analyzer.polarity_scores(sentence)
    score = vs['compound']
    if score >= 0.5:
        return 'pos'
    elif (score > -0.5) and (score < 0.5):
        return 'neu'
    elif score <= -0.5:
        return 'neg'

# 2. Target


def target(df):
    df['h'] = np.round(df.Upvote/(df.Upvote+df.Downvote), 2)
    return df

# 3. Drop Unwated Columns


def drop_cols(df):
    drop = ["Sum_of_Up_Down", "Upvote", "Downvote"]
    df = df.drop(drop, axis=1)
    return df

# 4. Number of sentence


def num_sentence(text):
    # return len(nltk.sent_tokenize(text))
    doc = nlp(text)
    return len(list(doc.sents))

# 5. Counter Upper Case Words


def count_upper(text):
    count = 0
    for i in text.split():
        if text.isupper():
            count += 1
    return count

# 6. Count Proper


def count_proper(text):
    count = 0
    for i in text.split():
        if text.istitle():
            count += 1
    return count

# 7. Count Emoji


def emoji_count(text):
    return emoji.emoji_count(text)

# 8. Remove Emoji


def remove_emoji(text):
    return text.encode('ascii', 'ignore').decode('ascii').strip()

# 9. Remove Punctuations


def remove_punctuations(text):
    return re.sub('[^\w\s%,-.]', "", text).strip()

# 10. Add POS tag for each word


def pos_tag(text):
    doc = nlp(text)
    return ' '.join([token.pos_ for token in doc])

# 11. Percentage of Nouns


def Noun(text):
    text_len = len(text.split())
    noun_count = 0
    for word in text.split():
        if word == 'NOUN':
            noun_count += 1
    return np.round((noun_count/text_len)*100, 2)

# 12. Percentage of Verb


def Verb(text):
    text_len = len(text.split())
    verb_count = 0
    for word in text.split():
        if word == 'VERB':
            verb_count += 1
    return np.round((verb_count/text_len)*100, 2)

# 13. Percentage of Adverb


def Adverb(text):
    text_len = len(text.split())
    adv_count = 0
    for word in text.split():
        if word == 'ADV':
            adv_count += 1
    return np.round((adv_count/text_len)*100, 2)

# 14. Percentage of Adjective


def Adj(text):
    text_len = len(text.split())
    adj_count = 0
    for word in text.split():
        if word == 'ADJ':
            adj_count += 1
    return np.round((adj_count/text_len)*100, 2)
#*************************************************************************************#
# *******Main Function*******


def features(df):
    # Filtering Reviews which has Sum of Upvote and Downvote which is greater than 10
    df['Sum_of_Up_Down'] = df.Upvote-df.Downvote
    df = df[df.Sum_of_Up_Down > 10]

    # Adding New Sentiment Column by calling the function **sentimental_Score**
    df['Sentiment'] = df.Review_Text.apply(sentimental_score)
    # Creating target and dropping unwanted columns
    df = target(df)
    df = drop_cols(df)

    # Length Before
    df["Len_before"] = df.Review_Text.apply(lambda x: len(x.split()))

    # Creating Num_Sentence
    df['Num_Sentence'] = df.Review_Text.apply(num_sentence)

    # Number of Question Mark
    df['No_QMark'] = df.Review_Text.str.count(pat='\?')

    # Number of Exclamatio Mark
    df['No_ExMark'] = df.Review_Text.str.count(pat='!')

    # Number of Upper Case Text
    df['No_Upper'] = df.Review_Text.apply(count_upper)

    # Number of Proper Case Text
    df['No_proper'] = df.Review_Text.apply(count_proper)

    # Count of Emoji
    df['Emoji_Count'] = df.Review_Text.apply(emoji_count)

    # Handling Emoji in review_text
    df['Review_Text'] = df.Review_Text.apply(remove_emoji)

    # Remove Punctuations
    df.Review_Text = df.Review_Text.apply(remove_punctuations)

    # Removed spell correction because its taking time in TextBlob

    # Apply Lemmatization for the review and remove stop words
    df.Review_Text = df.Review_Text.apply(lambda text: " ".join(token.lemma_ for token in nlp(text)
                                                                if not token.is_stop))

    # Length of the Review After removing stop words
    df["Len_after"] = df.Review_Text.apply(lambda x: len(x.split()))

    # Applying POS for all words
    df['POS'] = df.Review_Text.apply(pos_tag)

    # To avoid Zero Division Error
    df = df[df.Len_after >= 1]

    # Percentage of Noun
    df['Perc_Noun'] = df.POS.apply(Noun)

    # Percentage of Verb
    df['Perc_Verb'] = df.POS.apply(Verb)

    # Percentage of Adverb
    df['Perc_Adverb'] = df.POS.apply(Adverb)

    # Percentage of Adjective
    df['Perc_Adj'] = df.POS.apply(Adj)

    return df

#=================================================================================#
# Creates Predictors


def predictor(df, n=1):
    '''
    Pass the df for which important features with tfidf is needed
    unigrams occuring less than 1% of the time is not considered.
    list of words mentioned as stop words under tfidf is removed
    '''
    tfidf = TfidfVectorizer(
        token_pattern='(?ui)\\b\\w*[a-z]+\\w*\\b', min_df=0.01, stop_words="english")
    Matrix = tfidf.fit_transform(df.Review_Text)
    unigram = pd.DataFrame(Matrix.toarray(), columns=tfidf.get_feature_names())
    df_features = df.drop(['Review_Title', 'Review_Text', 'POS', 'Sentiment'], axis=1)
    main = unigram.join(df_features)
    main = main.fillna(0)
    X = main.drop('h', axis=1)
    y = main.h
    return X, y

#===============================================================================#
# Main Ranker Function

def rank(X,y):
    
    #XGB Regressor
    xgb = XGBRegressor(n_estimators = 1000,n_jobs=-1,random_state = 0)
    xgb.fit(X,y)
    # Predicting on test data
    y_pred = xgb.predict(X)
    
    return y_pred

##################################################################################
#

# In[68]:


#https://www.flipkart.com/redmi-8-ruby-red-64-gb/p/itmef9ed5039fca6?pid=MOBFKPYDCVSCZBYR&fm=organic&ssid=gcp7sydlxc0000001587014245662


# ## Things to do

# - Handle None in Location (Make it cwd)
# - Write doc string for all main function
# - ElementClickInterceptedException

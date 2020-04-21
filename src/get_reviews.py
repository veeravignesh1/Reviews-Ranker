from selenium import webdriver
from selenium.webdriver import Chrome, ChromeOptions
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.common.by import By
from selenium.common.exceptions import TimeoutException, WebDriverException
import os
import pandas as pd
import numpy as np



def get_review(user_url):

    '''Extracts reviews from user given `flipkart` product page and returns a `pandas dataframe`.

    Parameters
    -----------
    url: Product for which user wants to extract the review
    pages: Number of Pages of reviews the user likes to extract.By default `get_review`
    extracts 25 pages or 250 reviews

    Example
    -------
    >>> df=get_review("https://www.flipkart.com/redmi-8-ruby-red-64-gb/p/itmef9ed5039fca6?pid=MOBFKPYDCVSCZBYR")
    '''


    global product_name
    pages = 2  # change back to 25
    # User entered url
    url = user_url
    
    if 'flipkart' in url:
        review_url = url.replace('/p/', '/product-reviews/')

    # Driver essential to run automated chrome window
    chrome_options=ChromeOptions()
    chrome_options.binary_location = os.environ.get("GOOGLE_CHROME_BIN")
    chrome_options.add_argument("--headless")
    chrome_options.add_argument("--disable-dev-shm-usage")
    chrome_options.add_argument("--no-sandbox")
    # No option because its in currdir
    driver = Chrome(executable_path=os.environ.get("CHROMEDRIVER_PATH"),chrome_options=chrome_options)

    #List for Data
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
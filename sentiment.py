# ------------------------------------------------------------------------------
# This code consists of the sentiment analysis, language translation, spelling correction and language detection 
# ------------------------------------------------------------------------------

# ------------------------------------------------------------------------------
# List imports:
# ------------------------------------------------------------------------------
import pandas as pd 
import matplotlib.pyplot as plt 
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import LancasterStemmer, WordNetLemmatizer, PorterStemmer
from wordcloud import WordCloud, STOPWORDS 
from textblob import TextBlob 


# ------------------------------------------------------------------------------
# Import .csv file from our local directory
# ------------------------------------------------------------------------------
amz_reviews = pd.read_csv("/home/shaileshsarda/Desktop/ML Project/Sentiment Analysis/1429_1.csv",  low_memory=False)


# ------------------------------------------------------------------------------
# View  the loaded data
# ------------------------------------------------------------------------------
#print(amz_reviews)

# ------------------------------------------------------------------------------
# Check the dimensions of the data
# ------------------------------------------------------------------------------
print(amz_reviews.shape) # (34660, 21) = 34660  rows and 21 columns/variables/attributes 



# ------------------------------------------------------------------------------
# Check the list of columns
# ------------------------------------------------------------------------------
print(amz_reviews.columns)

"""The list of columns are : ['id', 'name', 'asins', 'brand', 'categories', 'keys', 'manufacturer',
       'reviews.date', 'reviews.dateAdded', 'reviews.dateSeen',
       'reviews.didPurchase', 'reviews.doRecommend', 'reviews.id',
       'reviews.numHelpful', 'reviews.rating', 'reviews.sourceURLs',
       'reviews.text', 'reviews.title', 'reviews.userCity',
       'reviews.userProvince', 'reviews.username'] """


# ------------------------------------------------------------------------------
# There are so many columns which are not useful for our sentiment analysis and it’s better to remove these columns. 
# There are many ways to do that: either just select the columns which you want to keep or select the columns you want to remove and then use the drop function to remove it from the data frame. 
# I prefer the second option as it allows me to look at each column one more time so I don’t miss any important variable for the analysis.
# ------------------------------------------------------------------------------


columns = ['id','name','keys','manufacturer','reviews.dateAdded', 'reviews.date','reviews.didPurchase',
          'reviews.userCity', 'reviews.userProvince', 'reviews.dateSeen', 'reviews.doRecommend','asins',
          'reviews.id', 'reviews.numHelpful', 'reviews.sourceURLs', 'reviews.title']

df = pd.DataFrame(amz_reviews.drop(columns,axis=1,inplace=False))


# ------------------------------------------------------------------------------
# Check the remain columns
# ------------------------------------------------------------------------------
print(df.columns) # the remian columns are : ['brand', 'categories', 'reviews.rating', 'reviews.text',      'reviews.username']


# ------------------------------------------------------------------------------
# Visualize the distributions of the ratings
# ------------------------------------------------------------------------------

df['reviews.rating'].value_counts().plot(kind='pie')


# ------------------------------------------------------------------------------
# Data pre-processing for textual variables
# ------------------------------------------------------------------------------


# ------------------------------------------------------------------------------
# Change the reviews type to string
# ------------------------------------------------------------------------------
df.dtypes  # Check the dataframe variabel type

df['reviews.text'] = df['reviews.text'].astype(str)


## Before lowercasing 
df['reviews.text'][2]


# ------------------------------------------------------------------------------
## Lowercase all reviews
# ------------------------------------------------------------------------------
df['reviews.text'] = df['reviews.text'].apply(lambda x: x.lower())
df['reviews.text'][2] # Just for checking the effect


# ------------------------------------------------------------------------------
# Remove Special characters
# ------------------------------------------------------------------------------
df['reviews.text'] = df['reviews.text'].str.replace('[^\w\s]', '')



# ------------------------------------------------------------------------------
# Remove stopwords
# ------------------------------------------------------------------------------
stopwords = stopwords.words('english')
df['reviews.text'] = df['reviews.text'].apply(lambda x: " ". join(x for x in x.split() if x not in stopwords))

# ------------------------------------------------------------------------------
# Stemmming
# ------------------------------------------------------------------------------
st = PorterStemmer()
df['reviews.text'] = df['reviews.text'].apply(lambda x: " ".join([st.stem(word) for word in x.split()]))


# ------------------------------------------------------------------------------
# Find / set Sentiment Score
# ------------------------------------------------------------------------------

def senti(x):
    return TextBlob(x).sentiment

df['senti_score'] = df['reviews.text'].apply(senti)


# ------------------------------------------------------------------------------
# correct spelling check using TextBlob
# ------------------------------------------------------------------------------

def correct_spell():
    input_stat = input("Enter Text here: ")
    blob_init = TextBlob(input_stat)
    return print(blob_init.correct())
correct_spell()



# ------------------------------------------------------------------------------
# Translate the sentance in different language like Hindi, Marathi, Urdu, Arabic, German, Spanish
# ------------------------------------------------------------------------------
def translate_stat():
    input_stat = input("Enter Text here: ")
    blob_init = TextBlob(input_stat)
    convert_to_language = input("Select any of language below to translate the sentance (base english):\n1. hi (Hindi)\n2. mr (Marathi)\n3. es (Spanish)\n4. de (German)\n5. ar (Arabic)\n6. ur (Urdu): \n>>>>")
    blob_init = blob_init.translate(from_lang="en", to='{}'.format(convert_to_language))
    return print(blob_init)
translate_stat()


# ------------------------------------------------------------------------------
# Detect the language type
# ------------------------------------------------------------------------------
def detect_lang():
    input_stat = input("Enter Text here: ")
    blob_init = TextBlob(input_stat)
    return blob_init.detect_language()
detect_lang()



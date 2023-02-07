#!/usr/bin/env python
# coding: utf-8

# In[3]:


import numpy as np
import pandas as pd


# In[4]:


df=pd.read_csv("C:/Users/20065/Downloads/stress.csv")
df.head()


# In[5]:


df.describe()


# In[6]:


#important step: check for null values 
df.isnull()


# In[7]:


df.isnull().sum() #instead of checking the whole data whether it is nll or not we can do this; if sum =0 means no null values


# In[8]:


#data we are dealing with is textual data; we should focus on the structure; various punctuation marks, 
#unwanted characters etc. will be there which we need to clean before summarizing it and generate clean results
#for cleaning we have a regression module in python
#stopward functionality showcases important words being used and removes common words like a, an, the etc.
#Snowball stemmer removes the prefixes like ing, ly etc; eg. caring becomes care etc. We do orphological affixation
#we do all this refinig of text data so that we can count the words and do accurate analysis
import nltk
import re
from nltk. corpus import stopwords
import string
nltk. download( 'stopwords' )
stemmer = nltk. SnowballStemmer("english")
stopword=set (stopwords . words ( 'english' ))
def clean(text):
    text = str(text) . lower()  #returns a string where all characters are lower case. Symbols and Numbers are ignored.
    text = re. sub('\[.*?\]',' ',text)  #substring and returns a string with replaced values.
    text = re. sub('https?://\S+/www\. \S+', ' ', text)#whitespace char with pattern
    text = re. sub('<. *?>+', ' ', text)#special char enclosed in square brackets
    text = re. sub(' [%s]' % re. escape(string. punctuation), ' ', text)#eliminate punctuation from string
    text = re. sub(' \n',' ', text)
    text = re. sub(' \w*\d\w*' ,' ', text)#word character ASCII punctuation
    text = [word for word in text. split(' ') if word not in stopword]  #removing stopwords
    text =" ". join(text)
    text = [stemmer . stem(word) for word in text. split(' ') ]#remove morphological affixes from words
    text = " ". join(text)
    return text
df [ "text"] = df["text"]. apply(clean)
 


# In[9]:


import numpy as np
from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator
import matplotlib.pyplot as plt
text = " ". join(i for i in df. text)
stopwords = set (STOPWORDS)
wordcloud = WordCloud( stopwords=stopwords,background_color="white") . generate(text)
plt. figure(figsize=(10, 10) )
plt. imshow(wordcloud)
plt. axis("off")
plt. show( )
#size of words in the image give idea on the frequency 


# In[10]:


from sklearn. feature_extraction. text import CountVectorizer
from sklearn. model_selection import train_test_split

x = np.array (df["text"])
y = np.array (df["label"])#0-no stress 1-stress

cv = CountVectorizer () #to map text and label #convert to vector #depends on frequency count of all the words
#fitting data is converting to vector so it can be mapped
X = cv. fit_transform(x) #fit and transform the text data to scale in standard deviation and mean format
#whenever working on functions we use mean and standard deviations mostly
print(X)
xtrain, xtest, ytrain, ytest = train_test_split(X, y,test_size=0.33)


# In[11]:


from sklearn.naive_bayes import BernoulliNB
model=BernoulliNB()
model.fit(xtrain,ytrain)
#o/p; BernoulliNB() implies our model is ready; all internal calculations are done


# In[19]:


user=input("Enter the text ")
data=cv.transform([user]).toarray()
output=model.predict(data)
print(output)


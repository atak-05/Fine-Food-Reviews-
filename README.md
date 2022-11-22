# Fine-Food-Reviews-
machine_learning

Within the project, a study was carried out on sentiment analysis in python. In this project, the data set called “Fine Food Reviews” in Amazon, which was taken from the kaggle library, was used as a data set. It was coded in python in the content of this project. Two techniques were used in this analysis. First is the use of the Vader model for the word bag. Secondly, the traditionally learned “embrace face” model, which is a roberta type model, was used.
This model is a more advanced transformer model. The results are matched between the two models. Also, the wrap face pipeline is used.

This study was conducted with the set of Good Food reviews on Amazon as a data set. In this data set, reviews about good food were collected. There are text reviews as well as 5 star ratings here. .


When starting this project, it was first worked on the kaggle notebook. And the relevant dataset is included in the project.


# 1. Notebook use and necessary libraries

In order to first read the data set from the notebook, pandas, one of the python libraries, as well as related libraries such as numpy, matplotlib and seaborn were included in the project.

Then the Natural Language Toolkit (NLTK) was used. NLTK is a platform for creating Python programs that work with human language data for application in statistical natural language processing (NLP). Tokenization, parsing, classification, stemming, tagging and semantic reasoning
Contains text manipulation libraries for execution.



```import numpy as np # linear algebra
import pandas as pd #data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns

plt.style.use('ggplot')
import nltk
```
# 2-Adding the Data Set to the Project and Examining the Data Set

The pandas library was used to read the csv file of the dataset to the project. The first 500 initial data of the Data Set are given.

```#Read in data
df = pd.read_csv('../input/amazon-fine-food-reviews/Reviews.csv')
print(df.shape)
df.head()
print(df.shape)
```

Here, each line has a unique Id, Product Id and User Id. In addition, each data has its own score score as Score. This score is the score given for the product by the person who commented the product. It gives a star rating between one and five. There are half a million reviews on this dataset for sentiment analysis here. However, first, the data set was sampled in this way.

```
df.head()
```
With the head command, we can see and reference which columns we have.
Now a bit of a quick "eda" is done to get an idea of what this dataset looks like. This eda tool is a tool created in python language and used for summarizing data. This "score" column, which we know is a value between one and five in the data set, will be used and a value counted on it will be made.
The star score here will be converted to a numerical score and displayed in the form of a bar graph. This chart shows the number of reviews by stars.


# Results on 3-Star Reviews
```
ax =  df['Score'].value_counts().sort_index()\
.plot(kind='bar',
      title='Count of Reviews by Stars',
      figsize=(10,5))

ax.set_xlabel('Review Stars')
plt.show()

```
# Basic NLTK
```
example = df['Text'][50]
print(example)
```
In the code sequence shown below, the tokenizer tool from the nltk model we previously imported was run for the example variable assigned in the previous code directory. And the end below. observed. According to this result, all the words in the comment were separated one by one.

This oatmeal is not good. Its mushy, soft, I don't like it. Quaker Oats is the way to go.

```
tokens= nltk.word_tokenize(example)
tokens[:10]
```

It is the process of marking words in text format for a specific part of a conversation with pos tags, another nltk tool in our project. Some examples of NLTK POS labeling are: CC, CD, EX, JJ, MD, NNP, PDT, PRP$, TO, etc. The POS tagger is used to assign grammatical information for each word of the sentence.

If this tool works for the specified example, the labels are as in the image below. In this way, each word is labeled.

```
tagged = nltk.pos_tag(tokens)
tagged[:10]
```

Here we see the top 10 tags for the sample comment. In fact, every label or abbreviation has a meaning. It is explained as Singular, Plural, Noun, Adjective etc.
```
tagged = nltk.pos_tag(tokens)
tagged[:10]
```

In this process, we placed the tags inside the comment as if they were part of the comment.
And it is assigned as entities value. This is done with the chuck method, which is one of the nltk tools. With this method, the tokenizer adds the results in the labeling to the results as a forename.

```
entities = nltk.chunk.ne_chunk(tagged)
entities.pprint()
```
So far, studies have been done on the nltk model. Now, studies on sentiment analysis will begin. For this, first the Vader model will be used. If this model is to be explained, the Vader model means VADER, Valence Aware Dictionary and SEntiment Reasoner. It is a Lexicon and rules-based sentiment analysis library. The library is popular in Sentiment Analytics. VADER is a simple rule-based sentiment analyzer. (magazinepark #)

In fact, the Vader model makes positive, neutral and negative inferences of certain words as a result of certain mathematical equations from existing words.

One of the disadvantages of this model is that it does not take into account the connection between conversations. Therefore, he cannot make an inference appropriate to the course of the sentence. After the theoretical explanation of the Vader model, it can now be put into practice.
First, the necessary imports for model use were added to the project.
The second tqdm model is included in the project to use a progress bar only when running the dataset. Later, the Sentiment Intensity Analyzer function that we imported was created as an object called sia.


#5-Vader Model and Sentiment Score and Analysis
```
from nltk.sentiment import SentimentIntensityAnalyzer
from tqdm.notebook import tqdm

sia = SentimentIntensityAnalyzer()
```

There is a Sentiment Intensity Analyzer object that we can run and see on this text.
In this Vader approach where emotion depends on words, “I am so happy!” When the sentence is analyzed by the Vader model, it is seen that it gives a positive result. Vader has calculated the negativity as zero for this word. And there is also the compund value. This value represents how much negative it is from negative to positive.
It can be said that the Vader model gives a correct result here.
```
sia.polarity_scores('I am so bad')
```
If the opposite is tried, if the sentence sia.polarity_scores(''this is the worst thing ever'') is analyzed, it is seen that the result is mostly negative.
```
sia.polarity_scores('I am so happy')
```



so if run polarity score on whole dataset with this code block
A loop was created and this analysis was run for 500 comments.
It is assigned a value named “res” for the results of this analysis.

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

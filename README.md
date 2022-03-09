# ![](https://ga-dash.s3.amazonaws.com/production/assets/logo-9f88ae6c9c3871690e33280fcf557f33.png) 
# Reddit Web Scraping and NLP Classification
#### <font color='darkgray'>Katherine Hickok</font>
#### <font color='darkgray'>General Assembly DSI Project 03</font>
#### <font color='darkgray'>January 18, 2022</font>

---
### PROBLEM STATEMENT

**Can we predict whether a post would be in the Witcher3 or NetflixWitcher subreddit better than the baseline accuracy?**
<br>

---
### BACKGROUND
Natural Langauge Processing (NLP) is a useful machine learning tool to help computers understand and model human language. ([source 01](https://www.ibm.com/cloud/learn/natural-language-processing)). This project used NLP tools and different classification models to detemine which subreddit a certain post would belong by analyzing the words in a post's title. The subreddits in question are associated with the Witcher universe: r/Witcher3 and r/NetflixWitcher. The Witcher was originally a Polish fantasy book series by Andrzej Sapkowski that has been developed into a series of video games (including the Witcher 3) by CD Projekt Red and a streaming tv series by Netflix ([source 02](https://witcher.fandom.com/wiki/The_Witcher_series)). Because one adaptation is a video game and the other is a tv series, it can be assumed that they will have similar terminology when it comes to character names and locations, yet some different terminology pertaining to game mechanics or actors/actresses in a television series. These differences in vocabulary should help the classification models determine if a post would belong in r/Witcher3 or r/NetflixWitcher.

Sources Cited: <br>
ibm: what is natural language processing? [source 01](https://www.ibm.com/cloud/learn/natural-language-processing)<br>
the witcher [source 02](https://witcher.fandom.com/wiki/The_Witcher_series) <br>
witcher 3 sureddit [source 03](https://www.reddit.com/r/witcher3) <br>
netflixwitcher subreddit [source 04](https://www.reddit.com/r/netflixwitcher) <br>
pros and cons of sentiment intensity analysis using vader [source 05](https://www.codeproject.com/Articles/5269447/Pros-and-Cons-of-NLTK-Sentiment-Analysis-with-VADE) <br>
ratio definition [source 06](https://www.dictionary.com/browse/ratio#:~:text=On%20the%20social%20media%20platform,and%20considering%20its%20content%20bad)
</br>

---

### DATASETS 

* [`witcher3_5000.pkl`](./dataframes/witcher3_5000.pkl): Witcher 3 Subreddit Posts (pickle dataframe) from witcher 3 sureddit [source 03](https://www.reddit.com/r/witcher3)
* [`netflixwitcher_5000.pkl`](./dataframes/netflixwitcher_5000.pkl): Netflix Witcher Subreddit Posts (pickle dataframe) from [source 04](https://www.reddit.com/r/netflixwitcher)
* [`all_witcher_5000.pkl`](./dataframes/all_witcher_5000.pkl): Combined Witcher 3 and Netflix Witcher Subreddit Posts (pickle dataframe)

The Witcher3 and NetflixWitcher dataframes are composed of cleaned data that was scraped from reddit using Pushshit's API. From each of the 5000 posts, the API scraped subreddit, title, selftext, score, upvote ratio, number of comments, and created utc (time the post was posted) information.

---

### ANALYSIS
#### WORD COUNTS
After vectorizing each subreddit dataframe, and identifying the top 15 words, two-word combinations, and three-word combinations by word count, there are words that are represntative of both subreddits, such as "witcher", "geralt" (character), "ciri" (character), and "kaer morhen" (location). There are also plenty of words that are solely indicative of Witcher 3 gameplay, including "gwent" (mini-game), "blood wine" (downloadable content), and "wolf school gear" (armor set). The r/NetflixWitcher subreddir contains words more associated with television vernacular, such as "spoilers" and "episode", and words characteristic of the plot, like "henry cavill" (actor who play Geralt) and "toss coin witcher" (song).

#### SENTIMENT INTENSITY ANALYSIS
Sentiment Intensity Analysis attempts to determine if the emotion of specific piece of text is considered to be negative, positive, or neutral. Some limitations of this specific analysis is that it sometimes fails to understand sarcasm, misspellings, and meme jargon as positive instead of negative or vice versa. ([source 05](https://www.codeproject.com/Articles/5269447/Pros-and-Cons-of-NLTK-Sentiment-Analysis-with-VADE)). This project used VADER to analyze sentiment intensity and a normalized compound score for each post was used to determine if a post was very positive, positive, neutral, negative, or very negative. Using this rating in conjuction with a post being considered controversial on reddit (upvote_ratio <= 0.5) or ratioed (term used on Twitter when a tweet has more comments than likes, aka a controversial tweet), can isolate posts that that were deemed controversial through multiple metrics. ([source 06](https://www.dictionary.com/browse/ratio#:~:text=On%20the%20social%20media%20platform,and%20considering%20its%20content%20bad)). Often these posts were misclassified, yet this tool identified posts of a derogatory nature concerning displeasure at casting non-white actors and actresses in the tv series in posts on the r/NetflixWitcher subreddit. 

#### MODELS
Different classification models were developed to predict which subreddit, a specifc title would belong. Below is a table that shows models that were developed using different NLP vectorization tools and classification models. Each model had specific hyperparameters tuned using either GridSearchCV or RandomizedSearchCV to identify which hyperparameters were more successful. Their corresponding train R^2 score and test R^2 score, are also displayed to show model effectiveness. The baseline accuracy of the model is 0.508.

|MODEL|TYPE|CHOSEN HYPERPARAMETERS|TRAIN R^2 SCORE|TEST R^2 SCORE|
|---|---|---|---|---|
|**Model 1**|CV/ MNB/ RS|'cv__max_features': 2000, 'cv__min_df': 0.0008283175685403598, 'cv__ngram_range': (1, 3), 'cv__stop_words': 'english'|0.838|0.812|
|**Model 2**|CV/ MNB/ GS|'cv__max_features': 750, 'cv__min_df': 5, 'cv__ngram_range': (1, 1), 'cv__stop_words': custom_stop_words|0.829|0.806|
|**Model 3**|CV/ LOG/ GS|'cv__max_features': 1000, 'cv__min_df': 5, 'cv__ngram_range': (1, 1), 'cv__stop_words': 'english', 'log__C': 1.0023052380778996, 'log__max_iter': 1000, 'log__penalty': 'l1', 'log__solver': 'liblinear'|0.855|0.816|
|**Model 4**|TFIDF/ LOG/ GS|log__C': 1.0023052380778996, 'log__max_iter': 1000, 'log__penalty': 'l2', 'log__solver': 'liblinear', 'tfidf__max_df': 0.25, 'tfidf__max_features': 1000, 'tfidf__min_df': 5, 'tfidf__ngram_range': (1, 1), 'tfidf__stop_words':custom_stop_words|0.858|0.811|
|**Model 5**|TFIDF/ MNB/ GS|'mnb__alpha': 1, 'tfidf__max_features': 1000, 'tfidf__min_df': 1, 'tfidf__ngram_range': (1, 1), 'tfidf__stop_words': custom_stop_words|0.843|0.804|
|**Model 6**|TFIDF/ MNB/ RS|'mnb__alpha': 1, 'tfidf__max_features': 1000, 'tfidf__min_df': 0.001989744173979979, 'tfidf__ngram_range': (1, 1), 'tfidf__stop_words': custom_stop_words|0.809|0.789|
|**Model 7**|CV/ RFC/ GS|'max_depth': None, 'min_samples_split': 10, 'n_estimators': 135|0.960|0.830|
<br>
CV- CountVectorizer, MNB- MultinomialNB, RS- RandomizedSearchCV, GS- GridSearch, LOG- LogisticRegession, TFIDF- TfidfVectorizer, RFC- RandomForestClassifier
</br>
<br>
Based on these results, the most successful model was Model 7 (test R^2 score: ± 0.830), which consisted of a RandomForestClassifier model modeling on count-vectorized training data. It was overfit, but still performed the best on the unknown test data. Model 7 had 417 misclassified posts but was generally better at predicting posts in r/Witcher3 (specificity: 0.867) as opposed to r/Netflix/Witcher (sensitivity: 0.791).
</br>
---

### CONCLUSIONS AND FURTHER RESEARCH
According to the problem statement, the all models tested were successful at beating the baseline score of 0.508 with the best model (Model 7) having a test R^2 score of 0.830 (1.6 times greater than the baseline). The top models were better at predicting the negative class (r/Witcher3) than the positive class (r/NetflixWitcher) aka higher specificity than sensitivity. 

Future research would involve improving models through more extensive hyperparameter tuning or looking at other classification models, such as K-Nearest Neighbors to determine which model has the best testing R^2 score. Also, understanding how the model misclassified data could improve model performance, however this may not solely be the model's fault since people often post in the wrong subreddit or using non-characteristic terminology. 
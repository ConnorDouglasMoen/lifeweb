import numpy as np
import guidedlda
from sklearn.feature_extraction.text import CountVectorizer
import operator

# read in the necessary files
data = []
with open('forevermissed_story_final.txt', 'r') as rows:
    for r in rows:
        data.append(r.strip())
lw_data = open('lemma_lifeweb_text.txt', 'r').readlines() ### For testing the model on sample memorial text


# Form the Document Frequency Matrix
# X is a matrix of dimensions (num documents, length of vocab) where X[i,j] is the number of times word j appears in doc i
vectorizer = CountVectorizer()
X = vectorizer.fit_transform(data) ### Train model with this

vectorizer2 = CountVectorizer()
X_test = vectorizer2.fit_transform(lw_data) ### Test Document Frequency Matrix --> fit model to this after training

# freq is an array of ints where freq[j] is the frequency word j appeared across all documents
freq = np.ravel(X.sum(axis=0))

# extract vocab list from X
# get vocabulary keys, sorted by value
vocab = [v[0] for v in sorted(vectorizer.vocabulary_.items(), key=operator.itemgetter(1))]
fdist = dict(zip(vocab, freq))
word2id = dict((v, idx) for idx, v in enumerate(vocab))


# Guided LDA with seed topics.
### This is a list of lists, where each inner list are seed words for a given topic.
###   The four topics given are, in order: relationships, education, hobbies, work
###   To add another topic, just include a new list of relevant seed words (up to your discretion what they are, but they must be
###   present in your training data, otherwise an error will be thrown).
seed_topic_list = [['relationship', 'mother', 'father',  'husband', 'wife', 'partner', 'lover', 'child', 'children', 'daughter', 'boyfriend', 'girlfriend', 'fiance', 'dating', 'family', 'spouse', 'wedding', 'friend', 'friendship', 'sister', 'brother', 'marriage', 'kids', 'birth', 'home', 'household' ],
                   ['classmate', 'school', 'highschool', 'college', 'class', 'grade', 'degree', 'education', 'university', 'graduate', 'lesson','freshman', 'sophomore', 'senior', 'junior'],
                   ['music', 'arts', 'cooking', 'surfing', 'sports', 'basketball', 'football', 'baseball', 'guitar', 'bass', 'violin', 'piano', 'food','photography', 'fitness', 'exercise', 'travel', 'film', 'movie', 'comedy', 'joke', 'television', 'instrument', 'craft', 'kite', 'flying','jazz', 'band','musician', 'song', 'sing', 'perform', 'actor', 'religion', 'church', 'bike', 'camp', 'beach', 'politics', 'government','democrat', 'trip'],
                   ['career', 'work', 'commute', 'salary', 'office', 'company', 'boss', 'coworker', 'colleague', 'money', 'corporate', 'corporation', 'worker','business', 'farming', 'sales', 'salesman', 'conference']]

# Initialize Model
model = guidedlda.GuidedLDA(n_topics=4, n_iter=100, random_state=7, refresh=20)

seed_topics = {}
for t_id, st in enumerate(seed_topic_list):
    for word in st:
        seed_topics[fdist[word]] = t_id

# Train model on X 
model.fit(X, seed_topics=seed_topics, seed_confidence=0.05)

# This might be useful i forgot what it does
n_top_words = 20
topic_word = model.topic_word_
for i, topic_dist in enumerate(topic_word):
    topic_words = np.array(vocab)[np.argsort(topic_dist)][:-(n_top_words+1):-1]
    print('Topic {}: {}'.format(i, ' '.join(topic_words)))

# Test model on X_test
doc_topic = model.transform(X_test)

### This is just for visualizing the doc_topic object, which includes the probablity that each document (story)
### is related to each topic.
for i in range(20):
    print("top topic: {} ".format(doc_topic[i]))

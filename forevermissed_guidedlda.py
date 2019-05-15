import numpy as np
import guidedlda
from sklearn.feature_extraction.text import CountVectorizer
import operator

# X = guidedlda.datasets.load_data(guidedlda.datasets.NYT)
# vocab = guidedlda.datasets.load_vocab(guidedlda.datasets.NYT)

# read in the necessary files
data = []
with open('forevermissed_story_final.txt', 'r') as rows:
    for r in rows:
        data.append(r.strip())

# now form the document frequency matrix
vectorizer = CountVectorizer()
X = vectorizer.fit_transform(data) # X is a matrix of dimensions (num documents, length of vocab) where X[i,j] is the number of times word j appears in doc i
print(X[0:50])
freq = np.ravel(X.sum(axis=0)) # freq is an array of ints where freq[j] is the frequency word j appeared across all documents

# extract vocab list from X
# get vocabulary keys, sorted by value
vocab = [v[0] for v in sorted(vectorizer.vocabulary_.items(), key=operator.itemgetter(1))]
print(vocab[0:50])
fdist = dict(zip(vocab, freq)) # return same `format as nltk

# word2id = dict((v, idx) for idx, v in enumerate(vocab))

print(X.shape)

print(X.sum())
# Normal LDA without seeding
model = guidedlda.GuidedLDA(n_topics=5, n_iter=100, random_state=7, refresh=20)
model.fit(X)

topic_word = model.topic_word_
n_top_words = 8
for i, topic_dist in enumerate(topic_word):
    topic_words = np.array(vocab)[np.argsort(topic_dist)][:-(n_top_words+1):-1]
    print('Topic {}: {}'.format(i, ' '.join(topic_words)))


# Guided LDA with seed topics.
seed_topic_list = [['relationship', 'mom', 'dad', 'husband', 'wife', 'partner', 'lover', 'son', 'daughter', 'boyfriend', 'girlfriend', 'fiance', 'dating', 'family', 'spouse', 'wedding', 'friend', 'friendship', 'sister', 'brother', 'marriage'],
                   ['classmate', 'school', 'highschool', 'high school', 'college', 'class', 'grade', 'degree', 'education', 'university', 'graduated'],
                   ['music', 'arts', 'cooking', 'birding', 'surfing', 'sports', 'basketball', 'football', 'baseball', 'guitar', 'bass', 'violin', 'piano', 'food', 'photography', 'fitness', 'exercise', 'travel', 'film', 'movie'],
                   ['career', 'job', 'work', 'commute', 'salary', 'office', 'company', 'boss', 'coworker', 'colleague']]

print(seed_topic_list)

model = guidedlda.GuidedLDA(n_topics=5, n_iter=100, random_state=7, refresh=20)

seed_topics = {}
for t_id, st in enumerate(seed_topic_list):
    for word in st:
        seed_topics[fdist[word]] = t_id

model.fit(X, seed_topics=seed_topics, seed_confidence=0.15)

n_top_words = 10
topic_word = model.topic_word_
for i, topic_dist in enumerate(topic_word):
    topic_words = np.array(vocab)[np.argsort(topic_dist)][:-(n_top_words+1):-1]
    print('Topic {}: {}'.format(i, ' '.join(topic_words)))

# write a try except that can handle key errors for words that do not match a key in the dictionary





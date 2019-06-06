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

lw_data = open('lemma_lifeweb_text.txt', 'r').readlines() 

# now form the document frequency matrix
vectorizer = CountVectorizer()
X = vectorizer.fit_transform(data) # X is a matrix of dimensions (num documents, length of vocab) where X[i,j] is the number of times word j appears in doc i
print(X[0:50])
freq = np.ravel(X.sum(axis=0)) # freq is an array of ints where freq[j] is the frequency word j appeared across all documents

vectorizer2 = CountVectorizer()
X_test = vectorizer2.fit_transform(lw_data)

# extract vocab list from X
# get vocabulary keys, sorted by value
vocab = [v[0] for v in sorted(vectorizer.vocabulary_.items(), key=operator.itemgetter(1))]
# print(vocab[0:50])
fdist = dict(zip(vocab, freq)) # return same `format as nltk

word2id = dict((v, idx) for idx, v in enumerate(vocab))

def get_top_n_words(corpus, n=None):
    """
    List the top n words in a vocabulary according to occurrence in a text corpus.
    
    get_top_n_words(["I love Python", "Python is a language programming", "Hello world", "I love the world"]) -> 
    [('python', 2),
     ('world', 2),
     ('love', 2),
     ('hello', 1),
     ('is', 1),
     ('programming', 1),
     ('the', 1),
     ('language', 1)]
    """
    vec = CountVectorizer().fit(corpus)
    bag_of_words = vec.transform(corpus)
    sum_words = bag_of_words.sum(axis=0) 
    words_freq = [(word, sum_words[0, idx]) for word, idx in vec.vocabulary_.items()]
    words_freq =sorted(words_freq, key = lambda x: x[1], reverse=True)
    print(words_freq)
    return words_freq[:n]


print(get_top_n_words(data, 25))


print(X.shape)

print(X.sum())
# Normal LDA without seeding
# model = guidedlda.GuidedLDA(n_topics=4, n_iter=100, random_state=7, refresh=20)
# model.fit(X)

# topic_word = model.topic_word_
# n_top_words = 10
# for i, topic_dist in enumerate(topic_word):
#     topic_words = np.array(vocab)[np.argsort(topic_dist)][:-(n_top_words+1):-1]
#     print('Topic {}: {}'.format(i, ' '.join(topic_words)))


# Guided LDA with seed topics.
seed_topic_list = [['relationship', 'mother', 'father',  'husband', 'wife', 'partner', 'lover', 'child', 'children', 'daughter', 'boyfriend', 'girlfriend', 
					'fiance', 'dating', 'family', 'spouse', 'wedding', 'friend', 'friendship', 'sister', 'brother', 'marriage', 'kids', 'birth', 'home', 'household' ],
                   ['classmate', 'school', 'highschool', 'college', 'class', 'grade', 'degree', 'education', 'university', 'graduate', 'lesson',
                   'freshman', 'sophomore', 'senior', 'junior'],
                   ['music', 'arts', 'cooking', 'surfing', 'sports', 'basketball', 'football', 'baseball', 'guitar', 'bass', 'violin', 'piano', 'food', 
                   	'photography', 'fitness', 'exercise', 'travel', 'film', 'movie', 'comedy', 'joke', 'television', 'instrument', 'craft', 'kite', 'flying',
                   	 'jazz', 'band','musician', 'song', 'sing', 'perform', 'actor', 'religion', 'church', 'bike', 'camp', 'beach', 'politics', 'government', 
                   	  'democrat', 'trip'],
                   ['career', 'work', 'commute', 'salary', 'office', 'company', 'boss', 'coworker', 'colleague', 'money', 'corporate', 'corporation', 'worker',
                    'business', 'farming', 'sales', 'salesman', 'conference']]

# print(seed_topic_list)

model = guidedlda.GuidedLDA(n_topics=4, n_iter=100, random_state=7, refresh=20)

seed_topics = {}
for t_id, st in enumerate(seed_topic_list):
    for word in st:
        seed_topics[fdist[word]] = t_id

model.fit(X, seed_topics=seed_topics, seed_confidence=0.05)

n_top_words = 20
topic_word = model.topic_word_
for i, topic_dist in enumerate(topic_word):
    topic_words = np.array(vocab)[np.argsort(topic_dist)][:-(n_top_words+1):-1]
    print('Topic {}: {}'.format(i, ' '.join(topic_words)))

doc_topic = model.transform(X_test)
for i in range(20):
	print("top topic: {} ".format(doc_topic[i])) 
# for i in range(9):
# 	print("top topic: {} Document: {}".format(doc_topic[i].argmax(),
#                                                   ', '.join(np.array(vocab)[list(reversed(X[i,:]))[0:5]])))
#argsort(axis = 1)



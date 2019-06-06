import numpy as np
import guidedlda

def ldac2dtm(stream, offset=0):
    """Convert an LDA-C formatted file to a document-term array

    Parameters
    ----------
    stream: file object
        File yielding unicode strings in LDA-C format.

    Returns
    -------
    dtm : array of shape N,V

    Notes
    -----
    If a format similar to SVMLight is the source, an `offset` of 1 may be used.
    """
    doclines = stream

    # We need to figure out the dimensions of the dtm.
    N = 0
    V = -1
    data = []
    for l in doclines:
        l = l.strip()
        # skip empty lines
        if not l:
            continue
        unique_terms = int(l.split(' ')[0])
        term_cnt_pairs = [s.split(':') for s in l.split(' ')[1:]]
        for v, _ in term_cnt_pairs:
            # check that format is indeed LDA-C with the appropriate offset
            if int(v) == 0 and offset == 1:
                raise ValueError("Indexes in LDA-C are offset 1")
        term_cnt_pairs = tuple((int(v) - offset, int(cnt)) for v, cnt in term_cnt_pairs)
        np.testing.assert_equal(unique_terms, len(term_cnt_pairs))
        V = max(V, *[v for v, cnt in term_cnt_pairs])
        data.append(term_cnt_pairs)
        N += 1
    V = V + 1
    dtm = np.zeros((N, V), dtype=np.intc)
    for i, doc in enumerate(data):
        for v, cnt in doc:
            np.testing.assert_equal(dtm[i, v], 0)
            dtm[i, v] = cnt
    return dtm


def load_data():
    dataset_ldac_fn = '/Users/aea/Desktop/GuidedLDA/guidedlda/tests/nyt.ldac'
    return ldac2dtm(open(dataset_ldac_fn), offset=0)


def load_vocab():
    vocab_fn = '/Users/aea/Desktop/GuidedLDA/guidedlda/tests/nyt.tokens'
    with open(vocab_fn) as f:
        vocab = tuple(f.read().split())
    return vocab


def load_titles():
    titles_fn = '/Users/aea/Desktop/GuidedLDA/guidedlda/tests/nyt.titles'
    with open(titles_fn) as f:
        titles = tuple(line.strip() for line in f.readlines())
    return titles

X = load_data()
print(X[0:50])
vocab = load_vocab()
word2id = dict((v, idx) for idx, v in enumerate(vocab))


# X = guidedlda.datasets.load_data(guidedlda.datasets.NYT)
# vocab = guidedlda.datasets.load_vocab(guidedlda.datasets.NYT)
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
seed_topic_list = [['game', 'team', 'win', 'player', 'season', 'second', 'victory'],
                   ['percent', 'company', 'market', 'price', 'sell', 'business', 'stock', 'share'],
                   ['music', 'write', 'art', 'book', 'world', 'film'],
                   ['political', 'government', 'leader', 'official', 'state', 'country', 'american', 'case', 'law', 'police', 'charge', 'officer', 'kill', 'arrest', 'lawyer']]

model = guidedlda.GuidedLDA(n_topics=5, n_iter=100, random_state=7, refresh=20)

seed_topics = {}
for t_id, st in enumerate(seed_topic_list):
    for word in st:
        seed_topics[word2id[word]] = t_id

model.fit(X, seed_topics=seed_topics, seed_confidence=0.15)

n_top_words = 10
topic_word = model.topic_word_
for i, topic_dist in enumerate(topic_word):
    topic_words = np.array(vocab)[np.argsort(topic_dist)][:-(n_top_words+1):-1]
    print('Topic {}: {}'.format(i, ' '.join(topic_words)))







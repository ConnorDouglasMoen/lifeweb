# import the necessary modules
import re
import gensim
from gensim.utils import simple_preprocess
from gensim.parsing.preprocessing import STOPWORDS
from nltk.stem import WordNetLemmatizer, SnowballStemmer
from nltk.stem.porter import *
import numpy as np
np.random.seed(2018)

import nltk
nltk.download('words')
nltk.download('wordnet')


tribute_text = open('forevermissed_tribute_text.txt', 'r').readlines()
tribute_text = map(lambda s: s.strip(), tribute_text)
tribute_text = list(tribute_text)


story_text = open('forevermissed_story_text.txt', 'r').readlines()
story_text = map(lambda s: s.strip(), story_text)
story_text = list(story_text)


cleaned_story_text = []
cleaned_tribute_text= []

TAG_RE = re.compile(r'<[^>]+>')

def remove_tags(text):
    return TAG_RE.sub('', text)

for item in story_text:
    item = remove_tags(str(item))
    item = item.replace('\xa0', ' ')
    cleaned_story_text.append(item)

for item in tribute_text:
    item = remove_tags(str(item))
    item = item.replace('\n', ' ')
    cleaned_tribute_text.append(item)
    
# print(cleaned_tribute_text)


english_story_text = []
english_tribute_text = []

words = set(nltk.corpus.words.words())

for item in cleaned_story_text:
    # print(item)
    curr = " ".join(w for w in nltk.wordpunct_tokenize(item) \
                 if w.lower() in words or not w.isalpha())
    english_story_text.append(curr)

print(cleaned_story_text[0:10])
print(english_story_text[0:10])

stemmer = SnowballStemmer('english')

def lemmatize_stemming(text):
    return stemmer.stem(WordNetLemmatizer().lemmatize(text, pos='v'))

def preprocess(text):
    result = []
    for token in gensim.utils.simple_preprocess(text):
        if token not in gensim.parsing.preprocessing.STOPWORDS and len(token) > 3:
            result.append(token)
            # result.append(lemmatize_stemming(token))
    return result

final_tribute_output = []
final_story_output = []

for item in cleaned_tribute_text:
	# lemmatized_content = lemmatize_stemming(item)
	preprocessed_content = preprocess(item)
	output_content = " ".join(preprocessed_content)
	final_tribute_output.append(output_content)

for item in cleaned_story_text:
	# lemmatized_content = lemmatize_stemming(item)
	preprocessed_content = preprocess(item)
	output_content = " ".join(preprocessed_content)
	final_story_output.append(output_content)

# print(final_story_output)

# output to different text files
with open('lemma_forevermissed_tribute_final.txt', 'w') as f:
    for item in final_tribute_output:
        f.write(str(item) + '\n')            
f.close()

with open('lemma_forevermissed_story_final1.txt', 'w') as f:
    for item in final_story_output:
        f.write(str(item) + '\n')           
f.close()







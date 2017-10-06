import xml.etree.cElementTree as ET
import nltk
import re

from HTMLParser import HTMLParser
from nltk.stem.snowball import SnowballStemmer
from gensim import corpora, models, similarities
from PIL import Image, ImageDraw 

transcript = 'power_of_introvert_transcript.xml'
tree = ET.parse(transcript)
root = tree.getroot()
h = HTMLParser()

stopwords = nltk.corpus.stopwords.words('english')
stopwords.extend([u"'s",u"n't",u"'m",u"'d"])
stemmer = SnowballStemmer("english")

def format_text(text):
    """Removes escape characters for text clustering"""
    text = text.replace("\n", " ")
    text = h.unescape(text)
    return text

def group_by_text(root, size):
    """Create text arrays based on the number of topics"""
    groups = []
    i = 0
    for i in xrange(len(root) - size):
        sentences = []
        duration = 0
        for j in xrange(size):
            sentences.append(format_text(root[i+j].text))
            duration += float(root[i+j].attrib["dur"])
        text = " ".join(sentences)
        group = {"start": float(root[i].attrib["start"]), "duration": duration, "text": text}
        groups.append(group)
    return groups 

def group_by_time(root, min_time):
    """Create text arrays based on duration"""
    groups = []
    i = 0
    for i in xrange(len(root)):
        sentences = []
        duration = 0
        for j in xrange(i, len(root)):
            sentences.append(format_text(root[j].text))
            duration += float(root[j].attrib["dur"])
            if (duration >= min_time):
                break
        text = " ".join(sentences)
        group = {"start": float(root[i].attrib["start"]), "duration": duration, "text": text}
        groups.append(group)
    return groups

def tokenize_and_stem(text):
    """Defines a tokenizer returning the set of stems in the text 
    while filtering out numeric tokens and punctuation"""
    tokens = [word for sentence in nltk.sent_tokenize(text) for word in nltk.word_tokenize(sentence)]
    filtered_tokens = []
    
    for token in tokens:
        if re.search('[a-zA-Z]', token):
            filtered_tokens.append(token)
    stems = [stemmer.stem(t) for t in filtered_tokens]
    return stems

def draw_topigram(corpus, ntopics, topic_order, lda, filename):
    w = len(corpus)
    h = 150
    bar_height = h / ntopics
    
    img = Image.new("RGB", (w, h), "black")
    draw = ImageDraw.Draw(img)
    
    for x in xrange(w):
        text_lda = lda[corpus[x]]
        print(text_lda)
        for t in text_lda:
            topic = topic_order.index(t[0])
            hue = int(360 * (topic * bar_height / float(h)))
            saturation = 100
            tint = int(pow(t[1], 1) * 100)
            color = "hsl("+ str(hue) + "," + str(saturation) + "%," + str(tint) + "%)"
            
            for y in xrange(topic * int(bar_height), (topic + 1) * int(bar_height)):
                draw.point((x, y), fill=color)
    
    img.save(filename, "PNG")

def build_topigram(root, time_clustering, size, ntopics, passes, filename):
    if(time_clustering):
        clusters = group_by_time(root, size) # size seconds long for each group
    else:
        clusters = group_by_text(root, size) # sentence length
        
    documents = []
    for cluster in clusters:
        documents.append(cluster["text"])
        
    tokenized_text = [tokenize_and_stem(text) for text in documents]
    
    sentences = [[word for word in text if word not in stopwords] for text in tokenized_text]
    
    dictionary = corpora.Dictionary(sentences)
    
    corpus = [dictionary.doc2bow(text) for text in sentences]
    
    lda = models.LdaModel(corpus, num_topics=ntopics, id2word=dictionary, passes=passes, chunksize=100)
    print lda
    print lda.show_topics(ntopics)
    
    max_topics = [(0,0,0)] * ntopics
    
    for x in xrange(len(corpus)):
        for y in lda[corpus[x]]:
            t = y[0]
            v = y[1]
            if max_topics[t][1] < v:
                max_topics[t] = (t,v,x)
    topic_order = sorted(max_topics, key = lambda s: s[2])
    topic_order = map(lambda s: s[0], topic_order)
    
    draw_topigram(corpus, ntopics, topic_order, lda, filename) 

filename = "topicogramtest.png"
build_topigram(root, True, 60, 10, 10, filename)

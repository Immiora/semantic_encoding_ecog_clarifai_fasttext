import matplotlib as mpl
mpl.use('TkAgg')
import matplotlib.pyplot as plt

##
import numpy as np
from PIL import Image
from gensim.models import KeyedVectors
import scipy.spatial

##
model = KeyedVectors.load_word2vec_format('./data/wiki.en.vec', binary=False)
# model = KeyedVectors.load_word2vec_format('./data/glove.6B.300d.vec', binary=False) # try glove, somewhat worse results

##
def look_up_word(w):
    ''' Look up one word in model '''
    return model.word_vec(w)

def look_up_by_n_grams(w):
    ''' Look up an out-of-vocabulary word by breaking it into 2 n-gram characters '''
    w_, w1, w2 = None, None, None
    for i in range(len(w)):
        try:
            w_ = (model.word_vec(w[:-i]) + model.word_vec(w[-i:]))/2
            w1, w2 = w[:-i], w[-i:]
            break
        except KeyError:
            pass
    return w_, w1, w2

def look_up_sentence(s):
    ''' Look up all words in a list '''
    s_ = []
    for w in s:
        w = w.lower()
        try:
            s_.append(look_up_word(w))
        except KeyError:
            try:
                if w.replace('-', '') in model.vocab:
                    w_ = look_up_word(w)
                elif len(w.replace('-', ' ').split(' ')) > 1:
                    ws = w.replace('-', ' ').split(' ')
                    w_ = np.sum([look_up_word(i) for i in ws], axis=0)/len(ws)
                    print('Token ' + w + ' broken into ')
                    print(ws)
                elif w == 'h2o':
                    w_ = look_up_word('water')
                else:
                    w_, w1, w2 = look_up_by_n_grams(w)
                    print('Token ' + w + ' not in vocabulary: broken into ' + w1 + ', ' + w2)
                s_.append(w_)
            except KeyError:
                print(w)
    return np.array(s_)

def plot_frame(f):
    ''' Plot one frame with assigned clarifai concepts and probabilities '''
    img = Image.open(im_dir + 'frame' + str(f+1) + '.jpg')
    plt.figure(figsize=(28,16), dpi=80)
    plt.subplot(121)
    plt.imshow(img)
    plt.title('Frame ' + str(f), fontsize=20)
    plt.axis('off')
    plt.subplot(122)
    #[plt.text(0, 0.7-i*0.05, w, fontsize=12) for i, w in enumerate(words[f])]
    [plt.text(0, 0.7-i*0.05, w, fontsize=12) for i, w in enumerate(words[f][:15])]
    [plt.text(0.3, 0.7-i*0.05, w, fontsize=12) for i, w in enumerate(words[f][15:30])]
    [plt.text(0.6, 0.7-i*0.05, w, fontsize=12) for i, w in enumerate(words[f][30:45])]
    [plt.text(0.9, 0.7-i*0.05, w, fontsize=12) for i, w in enumerate(words[f][45:])]
    plt.axis('off')


## load data
# the output in './data/output_clarifai_raw.npz' was manually cleaned up and saved into './data/corrected_clarifai_concepts_129.npz'
wordlist = np.load('./data/corrected_clarifai_concepts_129.npz', allow_pickle=True)
words = wordlist['words']
concept_list = wordlist['vocab']
im_dir = '/home/julia/Documents/project_chill/data/chill_frames/'
n = len(words)

## plot example frames
for _ in range(3):
    frame = np.random.randint(1, n)
    plot_frame(frame)

## look up word vectors for all concepts
sentences_mean = []
nw = []
for f in range(n):
    s = look_up_sentence(words[f])
    nw.append(s.shape[0])
    sentences_mean.append(np.mean(s, axis=0))
sentences_mean = np.array(sentences_mean)

##
np.save('./data/fasttext_sentences_mean_from_clarifai_concepts_129.npy', sentences_mean)

## plot similarity matrix across frames
sim = np.zeros((sentences_mean[::20].shape[0], sentences_mean[::20].shape[0]))
for i, s1 in enumerate(sentences_mean[::20]):
    for j, s2 in enumerate(sentences_mean[::20]):
        sim[i, j] = 1 - scipy.spatial.distance.cosine(s1, s2)

##
plt.imshow(sim, aspect='auto')
plt.colorbar()

import matplotlib as mpl
mpl.use('TkAgg')
import matplotlib.pyplot as plt

##
from sklearn.decomposition import PCA
import numpy as np
from gensim.models import KeyedVectors
import scipy.spatial.distance
from PIL import Image
import pickle

##
model = KeyedVectors.load_word2vec_format('./data/wiki.en.vec', binary=False)

##
def calculate_sim_matrix(data):
    n = data.shape[0]
    sim = np.zeros((n, n))
    for i, s1 in enumerate(data):
        for j, s2 in enumerate(data):
            sim[i, j] = 1 - scipy.spatial.distance.cosine(s1, s2)
    return sim


## load data
im_dir = '/home/julia/Documents/project_chill/data/chill_frames/'
sentences = np.load('./data/fasttext_sentences_mean_from_clarifai_concepts_129.npy', allow_pickle=True)
dims = sentences.shape
n = sentences.shape[0]
sentences_r = sentences.reshape((n, -1))

##
#sentences_rz = (sentences_r - np.mean(sentences_r, 0, keepdims=True))/np.std(sentences_r, 0, keepdims=True)

##
## do pca
pca = PCA(random_state=100, n_components=50, whiten=False)
sentences_pca = pca.fit_transform(sentences_r.astype(np.float32))

## plot examples
frame = np.random.randint(0, n)
plt.figure()
img = Image.open(im_dir + 'frame' + str(frame + 1) + '.jpg')
plt.subplot(121)
plt.imshow(img)
plt.subplot(122)
plt.plot(sentences_pca[frame, :])

## inverse pca
sentences_r_ = pca.inverse_transform(sentences_pca)
sentences_ = sentences_r_.reshape(dims)


## compare original and inverted cocept assignments
frame = np.random.randint(0, n)
for word in range(20):
    print(model.similar_by_vector(sentences[frame, word])[0][0])
    print(model.similar_by_vector(sentences_[frame, word])[0][0])


## compare original and inverted correlations over frames
S  = calculate_sim_matrix(sentences_r[:100,:])
S_ = calculate_sim_matrix(sentences_r_[:100,:])

##
plt.subplot(121)
plt.imshow(S, aspect='auto')
plt.colorbar()
plt.subplot(122)
plt.imshow(S_, aspect='auto')
plt.colorbar()

##
np.save('./data/semantic_pcs_from_fasttext_sentences_mean_129_noz_pc50.npy', sentences_pca)
pickle.dump(pca, open('./data/semantic_pcs_from_fasttext_sentences_mean_129_noz_pc50.p', 'wb'))

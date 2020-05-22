import matplotlib.pyplot as plt
import glob
import numpy as np
from PIL import Image
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
import pickle
import pandas as pd

##
def flatten_list(l):
    return [item for sublist in l for item in sublist]

def get_pca_example_indices(data, pc):
    temp = np.sort(data[:, pc])
    low = np.percentile(temp, 20)
    high = np.percentile(temp, 80)
    d = np.copy(data[:, pc])
    indices = np.arange(d.shape[0])
    print('Low: ' + str(np.sum(d<low)), ', high: ' + str(np.sum(d>high)))
    return np.vstack([indices[d<low][np.random.permutation(np.sum(d<low))[:100]], indices[d>high][np.random.permutation(np.sum(d>high))[:100]]]).flatten()

def plot_pca_examples(data, example_indices):
    plt.figure(figsize=(26, 14))
    plt.gca().set_facecolor("black")
    g = np.dstack(np.meshgrid(np.linspace(0, 11, 20), np.linspace(10, 0, 10))).reshape(-1, 2)
    for i, ps in zip(example_indices, g):
        img = imdata[i, :, :, :]
        img = Image.fromarray(img)
        ab = AnnotationBbox(OffsetImage(img),
                            ps, xycoords="data", frameon=False, box_alignment=(0, 0))
        plt.gca().add_artist(ab)
    plt.xlim(0, 11.9)
    plt.ylim(0, 11.2)
    plt.xticks([])
    plt.yticks([])

def plot_pca_examples_words(data, example_indices):
    def add_data_values(ax):
        for p in ax.patches:
            ax.annotate(str(int(p.get_height())), (p.get_x() * 1.005, p.get_height() * 1.005))

    min_words = words[example_indices[:100]]
    max_words = words[example_indices[100:]]
    pmin = pd.DataFrame({'words': flatten_list(min_words)})
    pmax = pd.DataFrame({'words': flatten_list(max_words)})
    plt.figure(figsize=(30, 18))
    plt.subplot(211)
    pmin['words'].value_counts()[:30].plot(kind='bar', color='k', fontsize=12)
    add_data_values(plt.gca())
    plt.subplot(212)
    pmax['words'].value_counts()[:30].plot(kind='bar', color='k', fontsize=12)
    add_data_values(plt.gca())
    # plt.savefig('./pics/pca_dims/clarifai_pca_dim' + str(pc) + '_words.png', dpi=80)


## load data
image_names = glob.glob('./data/chill_frames/*.jpg')
image_names = [i.replace('\\', '/') for i in image_names]
frames = np.array([int(i.split('.')[1].split('frame')[2]) for i in image_names])
indx = np.argsort(frames)
image_names = [image_names[i] for i in indx]

##
D = np.load('./data/corrected_clarifai_concepts_129.npz', allow_pickle=True)
words = D['words']

##
imdata = []
for im in image_names:
    temp = Image.open(im)
    temp.thumbnail([100, 100])
    imdata.append(np.array(temp))
imdata = np.array(imdata)


## load pca
sentences = np.load('./data/fasttext_sentences_mean_from_clarifai_concepts_129.npy').reshape((9749, -1))
sentences_pca = np.load('./data/semantic_pcs_from_fasttext_sentences_mean_129_noz_pc50.npy')

## show image examples per pc
pc = 0
example_indices = get_pca_example_indices(data=sentences_pca, pc=pc)
plot_pca_examples(sentences_pca, example_indices)

# show label histograms per pc
plot_pca_examples_words(sentences_pca, example_indices)


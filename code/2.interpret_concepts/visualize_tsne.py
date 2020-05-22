import matplotlib.pyplot as plt
import numpy as np
import glob
from PIL import Image
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

from matplotlib.offsetbox import OffsetImage, AnnotationBbox
from scipy.spatial.distance import cdist
from lap import lapjv

## load image urls and fasttext vectors
image_names = glob.glob('./data/chill_frames/*.jpg')
image_names = [i.replace('\\', '/') for i in image_names]
frames = np.array([int(i.split('.')[0].split('frame')[2]) for i in image_names])
indx = np.argsort(frames)
image_names = [image_names[i] for i in indx]

##
data_vectors = np.load('./data/semantic_pcs_from_fasttext_sentences_mean_129_noz_pc50.npy', allow_pickle=True)
m = data_vectors.shape[0]

## select subset of data (400 images) at random
size = 20
n = size * size
indx = np.random.permutation(m)[:n]
image_names = [image_names[i] for i in indx]
data_vectors = data_vectors[indx, :]

## load subset of images, resize
imdata = []
for im in image_names:
    temp = Image.open(im)
    temp.thumbnail([100, 100])
    imdata.append(np.array(temp))
imdata = np.array(imdata)


## run tsne on fasttext pca vectors
embeddings = TSNE(init='pca',verbose=2, random_state=200).fit_transform(data_vectors)
embeddings -= embeddings.min(axis=0)
embeddings /= embeddings.max(axis=0)

## plot scatter t-sne
plt.figure(figsize = (17, 9))
plt.scatter(embeddings[:, 0], embeddings[:, 1], c=indx)
cb= plt.colorbar(fraction=0.05, pad = 0.0125)
plt.xticks([]); plt.yticks([])

# plot images as scatter t-sne
plt.figure(figsize=(24, 12))
plt.gca().set_facecolor("black")
for pos, img in zip(embeddings, imdata):
    ab = AnnotationBbox(OffsetImage(img), 0.03 + pos * 0.94, xycoords="axes fraction", frameon=False)
    plt.gca().add_artist(ab)
plt.xticks([]); plt.yticks([])

# make grid
grid = np.dstack(np.meshgrid(np.linspace(0, 1, size), np.linspace(0, 1, size))).reshape(-1, 2)
plt.figure(figsize=(9,9))
plt.scatter(grid[:,0], grid[:,1])

# run linear assignment problem solver
cost_matrix = cdist(grid, embeddings, "sqeuclidean").astype(np.float32)
cost_matrix = cost_matrix * (100000 / cost_matrix.max())
row_asses, col_asses, x = lapjv(cost_matrix)

# plot trajectories obtained with linear assignment solver
grid_jv = grid[x]
pp_cmap = plt.cm.get_cmap(plt.rcParams["image.cmap"])
plt.figure(figsize=(17, 9))
for start, end, t in zip(embeddings, grid_jv, indx):
    plt.arrow(start[0], start[1], end[0] - start[0], end[1] - start[1],
          head_length=0.005, head_width=0.005, color=pp_cmap(t / 9), alpha=0.5)
plt.colorbar(cb.mappable, fraction=0.05, pad = 0.0125)
plt.xticks([]); plt.yticks([])

# plot resulting t-sne grid
plt.figure(figsize=(20, 30))
plt.gca().set_facecolor("black")
for pos, img in zip(grid_jv, imdata):
    img = Image.fromarray(img).resize((40, 40), Image.ANTIALIAS)
    ab = AnnotationBbox(OffsetImage(img),
                        pos * (size - 1) * 100, xycoords="data", frameon=False, box_alignment=(0, 0))
    plt.gca().add_artist(ab)
plt.xlim(0, size * 100); plt.ylim(0, size * 100)
plt.xticks([]); plt.yticks([])

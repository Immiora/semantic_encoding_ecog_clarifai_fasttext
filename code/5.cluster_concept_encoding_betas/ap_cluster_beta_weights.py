import numpy as np
import pickle
import glob
import matplotlib.pyplot as plt
from bokeh.core.property.validation import validate
from scipy.io import savemat, loadmat
from sklearn.cluster import AffinityPropagation
from scipy.stats import spearmanr
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
from PIL import Image
from scipy import stats
from matplotlib import cm as colormaps

##
zscore = lambda r: (r - np.mean(r, 0, keepdims=True)) / np.std(r, 0, keepdims=True)

def regress_z(x, z):
    c = np.linalg.lstsq(np.c_[z, np.ones(z.shape[0])], x)[0]
    r = x - np.c_[z, np.ones(z.shape[0])].dot(c)
    return r.astype(np.float32)

def smooth_signal(y, n):
    box = np.ones(n)/n
    ys = np.convolve(y, box, mode='same')
    return ys

def calculate_num_derivative(y):
    size = len(y)
    x = np.linspace(0, size, size)
    res = (np.roll(y, -1) - np.roll(y, 1)) / (np.roll(x, -1) - np.roll(x, 1))
    res[0] = (y[1] - y[0]) / (x[1] - x[0])
    res[-1] = (y[size - 1] - y[size - 2]) / (x[size - 1] - x[size - 2])
    return res

def arcor(x, y):
    # n_channels is dim 1
    # column-wise correlation for x (n x m) and y (n x m), output is z (m x 1)
    return np.array(map(lambda x_, y_: np.corrcoef(x_, y_)[0, 1], x.T, y.T))

def load_imdata():
    image_names = glob.glob('./data/chill_frames/*.jpg')
    image_names = [i.replace('\\', '/') for i in image_names]
    frames = np.array([int(i.split('.')[1].split('frame')[2]) for i in image_names])
    indx = np.argsort(frames)
    image_names = [image_names[i] for i in indx]
    D = np.load('./data/corrected_clarifai_concepts_129.npz')
    words = D['words']

    imdata = []
    for im in image_names:
        temp = Image.open(im)
        temp.thumbnail([100, 100])
        imdata.append(np.array(temp))
    imdata = np.array(imdata)

    return words, imdata

def plot_cluster_examples(example_indices):
    plt.figure(figsize=(16, 11))
    plt.gca().set_facecolor("black")
    g = np.dstack(np.meshgrid(np.linspace(0, 12, 20), np.linspace(10, 0, 10))).reshape(-1, 2)
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

##
kfolds = 5
load_dir = './results/encoding/ridge_semantic_pc50_concepts129_accCV5_alphaCV5/audio_boxcar_regressed_hfb_shift320ms/'
acc = loadmat(load_dir + 'accuracy.mat')['acc']
pmask = loadmat(load_dir+'pmask_mean_over_folds_1e-03_bonf.mat')['pmask'].flatten().astype(np.bool)
fasttext = np.load('./data/fasttext_sentences_mean_from_clarifai_concepts_129.npy')
fasttext_pca = np.load('./data/semantic_pcs_from_fasttext_sentences_mean_129_noz_pc50.npy')
pca = pickle.load(open('./data/semantic_pcs_from_fasttext_sentences_mean_129_noz_pc50.p','rb'))
codes = loadmat('./data/37subjs_chill_electrodes.mat')['codes'].flatten()
labels = np.load('./data/37subjs_elabels.npy')
words, imdata = load_imdata()

label_names = np.loadtxt('./data/electrode_label_list.txt', dtype=np.str).tolist()
label_names = [i.split('.')[0] for i in label_names]

##
weights = []
for kfold in range(kfolds):
    model_recon = pickle.load(open(load_dir + 'model'+str(kfold)+'.p', 'rb'))
    weights.append(model_recon.coef_[:, :-1])

weights = np.mean(np.array(weights), axis=0)
wz = zscore(weights)
wz_pmask = wz[pmask]
wz_indices = np.where(pmask == True)[0]



############################################### CLUSTER WEIGHTS ###############################################

affinity = spearmanr(wz_pmask.T)[0]

##
pref_range = np.linspace(-2, 0, 100)
nc = []
for pc in pref_range:
    print(pc)
    w_clustering = AffinityPropagation(affinity='precomputed', preference=pc, damping=.5, convergence_iter=10)
    clusters = w_clustering.fit(affinity)
    if len(clusters.cluster_centers_indices_):
        nc.append(len(clusters.cluster_centers_indices_))
    else:
        nc.append([])
nc_ = np.array([i if np.isfinite(i) else np.nan for i in nc])

##
plt.figure()
plt.plot(nc_, c=(.3, .3, .3), linewidth=3)
plt.xticks(range(0, len(np.round(pref_range, 2))), np.round(pref_range, 2), rotation=45)
plt.gca().xaxis.set_ticks_position('none')
plt.tight_layout()

##
w_clustering = AffinityPropagation(affinity='precomputed', damping=.5, preference=np.min(affinity)-2, convergence_iter=10)
clusters = w_clustering.fit(affinity)
print(len(clusters.cluster_centers_indices_))

##
subj_dist = []
for ind in range(len(clusters.cluster_centers_indices_)):
    subj_dist.append(len(np.unique(codes[wz_indices[clusters.labels_ == ind]])))
print(subj_dist)



################################ PLOT CLUSTER TIME COURSES & FRAMES FOR PEAKS/DIPS #####################################

savedir = './pics/_encoding/cluster_betas/pca50/ap_clusters_spearman_affinity_14clusters/'

for cluster in range(len(clusters.cluster_centers_indices_)):
    scores = fasttext_pca.dot(wz_pmask[clusters.cluster_centers_indices_[cluster]][:,None]).flatten()

    d = np.mean(zscore(fasttext_pca.dot(wz_pmask[clusters.labels_==cluster].T)), axis=1)
    sem = np.std(zscore(fasttext_pca.dot(wz_pmask[clusters.labels_==cluster].T)), axis=1) / float(np.sqrt(np.sum(clusters.labels_==cluster)))

    plt.figure(figsize=(8, 2), dpi=160)
    plt.plot(d, 'k', linewidth=1)
    plt.fill_between(range(fasttext_pca.shape[0]), d + sem, d - sem, color='k', alpha=0.3)
    [plt.plot([750*i, 750*i], [-np.max([np.abs(np.min(zscore(scores))), np.max(zscore(scores))]),
                               np.max([np.abs(np.min(zscore(scores))), np.max(zscore(scores))])], '--k', linewidth=.3, alpha=.5) for i in range(13)]
    plt.title('Cluster ' + str(cluster))

    fmin = np.where(scores<np.percentile(scores, 20))[0][np.random.permutation(np.sum(scores<np.percentile(scores, 20)))][:100]
    fmax = np.where(scores>np.percentile(scores, 80))[0][np.random.permutation(np.sum(scores>np.percentile(scores, 80)))][:100]

    plot_cluster_examples(np.concatenate([fmin, fmax]))



########################################## PLOT CLUSTER PC/SUBJ/LABEL DISTRIB ##########################################

# got pc 0, 1, 2, 3 the min/max in paper will be flipped (people will be put on the max side), thus here cols_neg are
# # full colors for flipped dims, only pc 5 (faces) remains the same: cols_pos for face presense -> full color

cols_neg = np.array(['firebrick', 'royalblue', 'orange', 'green', 'thistle'])
cols_pos = np.array(['mistyrose', 'lavender', 'bisque', 'honeydew', 'rebeccapurple'])
lobes = {}
lobes['o'] = (.1, .3, .3, .7)
lobes['t'] = (.3, .3, .3, .55)
lobes['p'] = (.5, .3, .3, .4)
lobes['m'] = (.7, .3, .3, .25)
lobes['f'] = (.9, .3, .3, .1)
lobes['_'] = (.9, .9, .9, .4)
cols_labs = np.array([(), lobes['t'], lobes['_'], lobes['f'], lobes['o'], lobes['t'], lobes['f'],
                      lobes['t'], lobes['p'], lobes['t'], lobes['_'], lobes['_'], lobes['o'],
                      lobes['f'], lobes['o'], lobes['f'], lobes['t'], lobes['m'], lobes['t'],
                      lobes['f'], lobes['f'], lobes['f'], lobes['_'], lobes['m'], lobes['_'],
                      lobes['m'], lobes['p'], lobes['_'], lobes['f'], lobes['f'], lobes['p'],
                      lobes['t'], lobes['p'], lobes['t'], lobes['t']])
alpha = 1e-20

##
for cluster in range(len(clusters.cluster_centers_indices_)):
    print(cluster)
    scores = fasttext_pca.dot(wz_pmask[clusters.cluster_centers_indices_[cluster]][:,None]).flatten()
    fmin = np.where(scores<np.percentile(scores, 10))[0][np.random.permutation(np.sum(scores<np.percentile(scores, 10)))][:500]
    fmax = np.where(scores>np.percentile(scores, 90))[0][np.random.permutation(np.sum(scores>np.percentile(scores, 90)))][:500]

    ts, ps = np.zeros(5), np.zeros(5),
    for ipc, pc in enumerate(range(4) + [5]):#range(5):
        ts[ipc], ps[ipc] = stats.ttest_ind(fasttext_pca[fmax, pc], fasttext_pca[fmin, pc])

    cols = [cols_pos[ps<alpha][i] if np.sign(ival) > 0 else cols_neg[ps<alpha][i] for i, ival in enumerate(ts[ps<alpha])]

    # distribution over pcs
    plt.figure(figsize=(6, 6))
    plt.pie(np.hstack([np.sum(np.abs(ts[ps<alpha]/(pca.explained_variance_ratio_[range(4) + [5]][ps<alpha]*100))),
                       np.abs(ts[ps < alpha] / (pca.explained_variance_ratio_[range(4) + [5]][ps < alpha] * 100))]), colors=['w']+cols,
            radius=.65, wedgeprops={'width':.1, 'edgecolor':'w', 'linewidth':.5}, counterclock=False)
    plt.title('cluster ' + str(cluster))

    # distribution over rois
    bins, counts = np.unique(labels[wz_indices[clusters.labels_ == cluster]], return_counts=True)
    corder = np.argsort(cols_labs[bins])[::-1]
    plt.pie(np.hstack([np.sum(counts), counts[corder]]), colors=[(1, 1, 1, 1)] + list(cols_labs[bins][corder]),
            radius=.5, wedgeprops={'width': .1, 'edgecolor': 'w', 'linewidth': .5}, counterclock=False)
    plt.title('cluster ' + str(cluster))

    # distribution over subjects
    bins, counts = np.unique(codes[wz_indices[clusters.labels_ == cluster]], return_counts=True)
    plt.pie(np.hstack([np.sum(counts), counts]), colors=[(1, 1, 1, 1)] + [(.7, .7, .8)] * len(counts),
            radius=.35, wedgeprops={'width': .1, 'edgecolor': 'w', 'linewidth': .5}, counterclock=False)
    plt.title('cluster ' + str(cluster))



############################################### CHECK SUBJECT CONTRIBUTION PROPORTION ##################################
for cluster in range(len(clusters.cluster_centers_indices_)):
    bins, counts = np.unique(codes[wz_indices[clusters.labels_==cluster]], return_counts=True)
    print('cluster' + str(cluster))
    print('\tmax eles from single subject: ' + str(np.max(counts)) + ', all eles: ' + str(np.sum(counts))
          + ' proportion:' + str(np.max(counts)/(np.sum(counts)/100.)))



################################## PLOT CLUSTER TIME COURSES WITH PERMUTATION BASELINE #################################

savedir = './pics/_encoding/cluster_betas/pca50/ap_clusters_spearman_affinity_14clusters/'

# for cluster in range(len(clusters.cluster_centers_indices_)):
for cluster in range(len(clusters.cluster_centers_indices_)):
    scores = fasttext_pca.dot(wz_pmask[clusters.cluster_centers_indices_[cluster]][:,None]).flatten()
    
    d = np.mean(zscore(fasttext_pca.dot(wz_pmask[clusters.labels_==cluster].T)), axis=1)
    sem = np.std(zscore(fasttext_pca.dot(wz_pmask[clusters.labels_==cluster].T)), axis=1) / float(np.sqrt(np.sum(clusters.labels_==cluster)))

    n_perm = 10000
    scores_perm = np.zeros((scores.shape[0], n_perm))
    for iperm in range(n_perm):
        temp_labels = clusters.labels_.copy()
        perm = np.random.permutation(temp_labels.shape[0])
        perm_labels = temp_labels[perm]
        scores_perm[:, iperm] = np.mean(zscore(fasttext_pca.dot(wz_pmask[perm_labels == cluster].T)), 1)

    upper_perm = np.percentile(scores_perm, 97.5, axis=1)
    lower_perm = np.percentile(scores_perm, 2.5, axis=1)
    mid_perm = np.percentile(scores_perm, 50, axis=1)

    plt.figure(figsize=(8, 2), dpi=160)
    plt.plot(mid_perm, 'grey', linewidth=.5, alpha=0.3)
    plt.plot(upper_perm, 'grey', linewidth=.5, alpha=0.2, linestyle=':')
    plt.plot(lower_perm, 'grey', linewidth=.5, alpha=0.2, linestyle=':')
    plt.fill_between(range(fasttext_pca.shape[0]), upper_perm, lower_perm, color= 'grey', alpha=0.1)
    [plt.plot([750*i, 750*i], [-np.max([np.abs(np.min(zscore(scores))), np.max(zscore(scores))]),
                               np.max([np.abs(np.min(zscore(scores))), np.max(zscore(scores))])], '--k', linewidth=.3, alpha=.5) for i in range(13)]
    plt.title('Cluster ' + str(cluster))






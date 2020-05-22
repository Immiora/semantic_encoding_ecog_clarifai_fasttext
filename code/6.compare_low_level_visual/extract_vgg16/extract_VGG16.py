import chainer
import numpy as np
import matplotlib.pyplot as plt
import scipy.signal
from chainer.links import VGG16Layers
from PIL import Image
import json
import glob

def plot_frame(img):
    plt.figure(figsize=(8,8), dpi=80)
    plt.imshow(img)

def gauss_kern(size, sizey=None):
    """ Returns a normalized 2D gauss kernel array for convolutions """
    size = int(size)
    if not sizey:
        sizey = size
    else:
        sizey = int(sizey)
    x, y = np.mgrid[-size:size+1, -sizey:sizey+1]
    g = np.exp(-(x**2/float(size)+y**2/float(sizey)))
    return g / g.sum()


def blur_image(im, n, ny=None) :
    """ blurs the image by convolving with a gaussian kernel of typical
        size n. The optional keyword argument ny allows for a different
        size in the y direction.
    """
    g = gauss_kern(n, sizey=ny)
    improc = scipy.signal.convolve(im,g, mode='valid')
    return(improc)

def smooth_downsample_h(h):
    '''h per frame'''
    if h.shape[1] * h.shape[2] * h.shape[3] > 100000:
        p = np.ceil(np.sqrt(h.shape[1] * h.shape[2] * h.shape[3] / 100000.)).astype(np.int16)
        nx, ny = blur_image(h[0, 0], n=p * 0.5).shape
        y = np.zeros((h.shape[0], h.shape[1], nx, ny))
        for img in range(h.shape[0]):
            for filter in range(h.shape[1]):
                y[img, filter] = blur_image(h[img, filter], n=p * 0.5)
        return y[:, :, ::p, ::p]
    else:
        return h.squeeze()

##
with open('./code/make_baseline_vision/vgg16/imagenet_class_index.json') as json_file:
    labels = json.load(json_file)

##
model = VGG16Layers()

##
#f = 5400
#img = Image.open('/vol/ccnlab-scratch1/julber/clarifai_wordvec_decoding/data/chill_frames/frame'+str(f)+'.jpg')
#plot_frame(img)

## load data
indices = range(9749) # number of frames
image_names = glob.glob('./data/chill_frames/*.jpg')
image_names = [i.replace('\\', '/') for i in image_names]
frames = np.array([int(i.split('.')[1].split('frame')[2]) for i in image_names])
indx = np.argsort(frames)
image_names = np.array([image_names[i] for i in indx])

##
imdata = []
for im in image_names[indices]:
    temp = Image.open(im).resize((224, 224))
    imdata.append(np.array(temp))
imdata = np.array(imdata)

## memory errors if not splitting the data, running only a subset of frames at a time
l = 'pool1'
output = []
n = int(np.ceil(imdata.shape[0]/20.))
for i in range(n):
    st, en = 20*i, 20*i+20 if 20*i+20 < imdata.shape[0] else imdata.shape[0]
    print(indices[st], indices[en-1])
    with chainer.using_config('train', False):
        with chainer.using_config('enable_backprop', False):
            output.append(smooth_downsample_h(model.extract(images=imdata[st:en], layers=[l])[l].data))
            # label = np.argmax(model.predict(images=[img]).data) # verify that labels are correct


##
output=np.concatenate(output)
np.save('./data/vgg16/'+l+'_'+str(np.max(indices)+1)+'.npy', output)



## ######################### when all subsets of frames are finished: reload and concatenate output
a = np.load('./data/vgg16/'+l+'_1000.npy')
b = np.load('./data/vgg16/'+l+'_5000.npy')
c = np.load('./data/vgg16/'+l+'_9000.npy')
d = np.load('./data/vgg16/'+l+'_9749.npy')


## visualize representations per frame
plt.figure(figsize=(8, 8), dpi=80)
plt.subplot(121)
plt.imshow(Image.fromarray(imdata[9000, :, :, :]))
plt.subplot(122)
plt.imshow(d[0, 0])

## concatenate
temp = np.vstack([a, b, c, d])

## save
np.save('./data/vgg16/'+l+'.npy', temp.reshape((9749, -1)))



# original shapes after downsample_smooth:
# pool1: (9749, 64, 37, 37)
# pool2: (9749, 128, 18, 18)
# pool3: (9749, 256, 13, 13)
# pool4: (9749, 512, 6, 6)
# pool5: (9749, 512, 7, 7)
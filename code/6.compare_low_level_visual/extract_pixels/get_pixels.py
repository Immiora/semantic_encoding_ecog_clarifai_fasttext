import glob
import numpy as np
from PIL import Image

def remove_transparency(im, bg_colour=(255, 255, 255)):

    # Only process if image has transparency
    if im.mode in ('RGBA', 'LA') or (im.mode == 'P' and 'transparency' in im.info):

        # Need to convert to RGBA if LA format due to a bug in PIL
        alpha = im.convert('RGBA').split()[-1]

        # Create a new background image of our matt color.
        # Must be RGBA because paste requires both images have the same format

        bg = Image.new("RGBA", im.size, bg_colour + (255,))
        bg.paste(im, mask=alpha)
        return bg

    else:
        return im

## load data
image_names = glob.glob('./data/chill_frames/*.jpg')
image_names = [i.replace('\\', '/') for i in image_names]
frames = np.array([int(i.split('.')[1].split('frame')[2]) for i in image_names])
indx = np.argsort(frames)
image_names = [image_names[i] for i in indx]


## colored and grayscale pixels
imdata, greydata = [], []
for im in image_names:
    temp = Image.open(im)
    temp.thumbnail([100, 100])
    imdata.append(np.array(temp))
    temp2=remove_transparency(temp).convert('L')
    greydata.append(np.array(temp2))
imdata = np.array(imdata)
greydata = np.array(greydata)


##
imdata = imdata.reshape((imdata.shape[0], -1))
np.save('./data/chill_pxl_colored.npy', imdata)

greydata = greydata.reshape((greydata.shape[0], -1))
np.save('./data/chill_pxl_grey.npy', greydata)

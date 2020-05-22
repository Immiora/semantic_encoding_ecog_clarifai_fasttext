import matplotlib.pyplot as plt
from clarifai.rest import ClarifaiApp
from clarifai.rest import Video as ClVideo
from clarifai.rest import Image as ClImage
import time
import glob
import numpy as np

##
image_names = glob.glob('./data/chill_frames/*.jpg')
image_names = [i.replace('\\', '/') for i in image_names]
frames = np.array([int(i.split('.')[1].split('frame')[2]) for i in image_names])
indx = np.argsort(frames)
image_names = [image_names[i] for i in indx]

##
app = ClarifaiApp(api_key='...')
model = app.models.get("general-v1.3")

## predict with the model
concepts, probs = [], []


def get_frame_pred(im):
    a = model.predict([im])
    c = [str(i['name']) for i in a['outputs'][0]['data']['concepts']]
    p = [float(i['value']) for i in a['outputs'][0]['data']['concepts']]
    return c, p

##
for f in range(len(frames)):
    print('frame ' + str(f))
    for attempt in range(10): # sometimes connection times out, dirty fix
        try:
            im = ClImage(file_obj=open(image_names[f], 'rb'))
            out = get_frame_pred(im)
            concepts.append(c)
            probs.append(p)
        except:
            time.sleep(5)
        else:
            break
    else:
        break

##
np.save('./data/output_clarifai_raw.npz', concepts=concepts, probs=probs)

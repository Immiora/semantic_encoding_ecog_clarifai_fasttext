import numpy as np

a=np.load('./data/corrected_clarifai_concepts_129.npz')

##
bin = np.zeros((len(a['words']), len(a['vocab'])))

for i, concept in enumerate(a['vocab']):
    for frame, frame_words in enumerate(a['words']):
        if concept in frame_words:
            bin[frame, i] = 1

##
np.save('./data/corrected_clarifai_concepts_129_binary.npy', bin.astype(np.int32))
np.savez('./data/corrected_clarifai_concepts_129_binary.npz', vectors=bin.astype(np.int32), names=a['vocab'])

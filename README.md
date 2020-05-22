# semantic_encoding_ecog_clarifai_fasttext
Repository contains code for the results described in the paper

Here, we model the cortical responses to semantic information that was extracted from the visual stream of a feature film, employing artificial neural network models. The semantic representations are extracted from the film by combining perceptual ([Clarifai](https://www.clarifai.com/) model) and linguistic information ([Fasttext](https://fasttext.cc/) model). We test whether these representations are useful in studying the human brain data. 

For the neural encoding model we use electrocorticography (ECoG) responses to a short movie from 37 subjects. We fit ECoG cortical patterns across multiple regions using the semantic components extracted from the film frames. 

We found that individual semantic components reflected fundamental semantic distinctions in the visual input, such as presence or absence of people, human movement, landscape scenes, human faces, etc. Moreover, each semantic component mapped onto a distinct functional cortical network involving high-level cognitive regions in occipitotemporal, frontal and parietal cortices. 

The present work demonstrates the potential of the data-driven methods from information processing fields to explain patterns in the human brain responses. 

The repository contains code for

- extraction of the visual concepts using [Clarifai](https://www.clarifai.com/)
- extraction of the [Fasttext](https://fasttext.cc/) word embeddings per frame based on the visual concepts
- interpretation and visualization of the extracted concepts
- linear regression models to fit the extracted concepts to the ECoG responses
- affinity propagation clustering of the beta-weights of the regression for intepretation of the neural tuning profiles
- control linear models using low-level visual feature sets and binary labels of the visual concepts 

Dependencies:
- [Scikit-learn](https://scikit-learn.org/)
- [Numpy](https://numpy.org/)
- [Scipy](https://www.scipy.org/)
- [Clarifai API](https://github.com/Clarifai/clarifai-python)
- [Statsmodels](https://www.statsmodels.org/)

Originally written in Python 2.7, [Anaconda](https://www.anaconda.com/) release

![Alt text](/git_front.png?raw=true "Main results")

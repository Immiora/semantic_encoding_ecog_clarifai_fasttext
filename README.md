# semantic_encoding_ecog_clarifai_fasttext
Repository contains code for the results described in the paper


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

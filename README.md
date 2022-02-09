# Semantic Encoding of Clarify Visual Concepts + Fasttext Embeddings in ECoG

Repository contains code for the results described in the paper:
Berezutskaya, J., Freudenburg, Z. V., Ambrogioni, L., Güçlü, U., van Gerven, M. A., & Ramsey, N. F. (2020). [Cortical network responses map onto data-driven features that capture visual semantics of movie fragments](https://www.nature.com/articles/s41598-020-68853-y). Scientific reports, 10(1), 1-21.


### Content
The repository contains code for

- extraction of the visual concepts using [Clarifai](https://www.clarifai.com/)
- extraction of the [Fasttext](https://fasttext.cc/) word embeddings per frame based on the visual concepts
- interpretation and visualization of the extracted concepts
- linear regression models to fit the extracted concepts to the ECoG responses
- affinity propagation clustering of the beta-weights of the regression for intepretation of the neural tuning profiles
- control linear models using low-level visual feature sets and binary labels of the visual concepts 

### Dependencies
- [Scikit-learn](https://scikit-learn.org/)
- [Numpy](https://numpy.org/)
- [Scipy](https://www.scipy.org/)
- [Clarifai API](https://github.com/Clarifai/clarifai-python)
- [Statsmodels](https://www.statsmodels.org/)

Originally written in Python 2.7, [Anaconda](https://www.anaconda.com/) release

### Citation

If this code has been helpful to you, please cite the related paper:

```
@article{berezutskaya2020cortical,
  title={Cortical network responses map onto data-driven features that capture visual semantics of movie fragments},
  author={Berezutskaya, Julia and Freudenburg, Zachary V and Ambrogioni, Luca and G{\"u}{\c{c}}l{\"u}, Umut and van Gerven, Marcel AJ and Ramsey, Nick F},
  journal={Scientific reports},
  volume={10},
  number={1},
  pages={1--21},
  year={2020},
  publisher={Nature Publishing Group}
}
```

### Visual summary

![Alt text](/git_front.png?raw=true "Main results")

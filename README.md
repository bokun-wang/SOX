# Finite-Sum Coupled Compositional Stochastic Optimization: Theory and Applications

This repository contains the code on three tasks: average precision (AP) maximization, p-norm push (PnP) for bipartite ranking, and neighborhood component analysis (NCA) for unsupervised representation learning. Please refer to each folder for detailed instructions.

## How to cite
If you find our work useful, please consider citing [our paper](https://proceedings.mlr.press/v162/wang22ak.html)
```
@InProceedings{pmlr-v162-wang22ak,
  title = 	 {Finite-Sum Coupled Compositional Stochastic Optimization: Theory and Applications},
  author =       {Wang, Bokun and Yang, Tianbao},
  booktitle = 	 {Proceedings of the 39th International Conference on Machine Learning},
  pages = 	 {23292--23317},
  year = 	 {2022},
  editor = 	 {Chaudhuri, Kamalika and Jegelka, Stefanie and Song, Le and Szepesvari, Csaba and Niu, Gang and Sabato, Sivan},
  volume = 	 {162},
  series = 	 {Proceedings of Machine Learning Research},
  month = 	 {17--23 Jul},
  publisher =    {PMLR},
  pdf = 	 {https://proceedings.mlr.press/v162/wang22ak/wang22ak.pdf},
  url = 	 {https://proceedings.mlr.press/v162/wang22ak.html},
  abstract = 	 {This paper studies stochastic optimization for a sum of compositional functions, where the inner-level function of each summand is coupled with the corresponding summation index. We refer to this family of problems as finite-sum coupled compositional optimization (FCCO). It has broad applications in machine learning for optimizing non-convex or convex compositional measures/objectives such as average precision (AP), p-norm push, listwise ranking losses, neighborhood component analysis (NCA), deep survival analysis, deep latent variable models, etc., which deserves finer analysis. Yet, existing algorithms and analyses are restricted in one or other aspects. The contribution of this paper is to provide a comprehensive convergence analysis of a simple stochastic algorithm for both non-convex and convex objectives. Our key result is the improved oracle complexity with the parallel speed-up by using the moving-average based estimator with mini-batching. Our theoretical analysis also exhibits new insights for improving the practical implementation by sampling the batches of equal size for the outer and inner levels. Numerical experiments on AP maximization, NCA, and p-norm push corroborate some aspects of the theory.}
}

```

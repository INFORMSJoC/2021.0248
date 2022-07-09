[![INFORMS Journal on Computing Logo](https://INFORMSJoC.github.io/logos/INFORMS_Journal_on_Computing_Header.jpg)](https://pubsonline.informs.org/journal/ijoc)

# SLPMM

This archive is distributed in association with the [INFORMS Journal on
Computing](https://pubsonline.informs.org/journal/ijoc) under the [MIT License](https://github.com/INFORMSJoC/2021.0248/blob/master/LICENSE).

The software and data in this repository are a snapshot of the software and data
that were used in the research reported on in the paper 
[Solving Stochastic Optimization with Expectation Constraints Efficiently by a Stochastic Augmented Lagrangian-Type Algorithm](https://doi.org/10.1287/ijoc.2021.0248) by L. Zhang, Y. Zhang, J. Wu and X. Xiao. 


## Cite

To cite this software, please cite the [paper](https://doi.org/10.1287/ijoc.2021.0248) using its DOI and the software itself, using the following DOI.

[![DOI](https://zenodo.org/badge/285853815.svg)](https://zenodo.org/badge/latestdoi/285853815)

Below is the BibTex for citing this version of the code.

```
@article{SLPMM2022,
  author =        {Liwei Zhang, Yule Zhang, Jia Wu and Xiantao Xiao},
  publisher =     {INFORMS Journal on Computing},
  title =         {Solving Stochastic Optimization with Expectation Constraints Efficiently by a Stochastic Augmented {L}agrangian-Type Algorithm},
  year =          {2022},
  doi =           {10.5281/zenodo.3977566},
  url =           {https://github.com/INFORMSJoC/2021.0248},
}  
```

## Description

The goal of this software is to compare the performance of stochastic linearized proximal method of multipliers (**SLPMM**) proposed in the [paper](https://doi.org/10.1287/ijoc.2021.0248) with several existing algorithms for minimizing a convex expectation function subject to a set of inequality convex expectation constraints. A preprint of this paper is available on [arXiv](https://arxiv.org/abs/2106.11577).

Three numerical examples are tested in this software: 
- Neyman-Pearson classification, 
- stochastic quadratically constrained quadratic programming (QCQP), 
- second-order stochastic dominance (SSD)  constrained portfolio optimization.

This software contains three folders: `NP_classification`, `QCQP`, `SSD`. 
- `NP_classification`: solves Neyman-Pearson classification problems.
- `QCQP`: solves stochastic quadratically constrained quadratic programs.
- `SSD`: solves second-order stochastic dominance (SSD)  constrained portfolio optimization problems.

The tested existing algorithms include:
- **CSA**: Lan G, Zhou Z (2020) Algorithms for stochastic optimization with function or expectation constraints.
*Comput. Optim. Appl.* 76(2):461–498. [link](https://doi.org/10.1007/s10589-020-00179-x)
- **YNW**: Yu H, Neely MJ, Wei X (2017) Online convex optimization with stochastic constraints. *Advances in Neural
Information Processing Systems*, 1428–1438. [link](https://papers.nips.cc/paper/2017/hash/da0d1111d2dc5d489242e60ebcbaf988-Abstract.html)
- **PSG**: Xiao X (2019) Penalized stochastic gradient methods for stochastic convex optimization with expectation
constraints, *optimization-online*. [link](http://www.optimization-online.org/DB_HTML/2019/09/7364.html)
- **APriD**: Yan Y, Xu Y (2022) Adaptive primal-dual stochastic gradient method for expectation-constrained convex
stochastic programs. *Math. Program. Comput.* 14(2):319–363. [link](https://doi.org/10.1007/s12532-021-00214-w)
- **PALEM**: Dentcheva D, Martinez G, Wolfhagen E (2016) Augmented Lagrangian methods for solving optimization
problems with stochastic-order constraints. *Oper. Res.* 64(6):1451–1465. [link](https://doi.org/10.1287/opre.2016.1521)

This software had been carried out using MATLAB R2020a on a desktop computer with Intel(R) Xeon(R) E-2124G 3.40GHz and 32GB memory. The MATLAB function *refline* is required which is available in the Statistics Toolbox of MATLAB.


## Results

1. The files in folder `NP_classification/results` show the results of comparison between  **CSA**, **YNW**, **PSG**, **APriD** and **SLPMM** for Neyman-Pearson classification.
- Figure 1 in the paper shows the results of comparison of algorithms on dataset *gisette*.
- Figure 2 in the paper shows the results of comparison of algorithms on dataset *CINA*.
- Figure 3 in the paper shows the results of comparison of algorithms on dataset *MNIST*.

2. The files in folder `QCQP/results` show the results of comparison between   **YNW**, **PSG**, **APriD** and **SLPMM** for stochastic quadratically constrained quadratic programming.
- Figure 4 in the paper shows the results of comparison of algorithms on stochastic quadratically constrained quadratic programming.


3. The files in folder `SSD/results` show the results of comparison between   **YNW**, **PSG**, **APriD**, **PALEM** and **SLPMM** for SSD constrained portfolio optimization.
- Figure 5 in the paper shows the results of comparison of algorithms on SSD constrained portfolio optimization.

## Replicating

- To replicate the results in Figure 1-3, run the `NP_classification/test_NP_classification_logloss.m` script.
- To replicate the results in Figure 4, run the `QCQP/test_QCQP.m` script.
- To replicate the results in Figure 5, run the `SSD/test_portfolio_SSD.m` script.

## Remark
 The elapsed cpu time of the experiments is much longer than that is shown in the figures (pure time of the algorithms). The reason is that we have to track the true values of objective and constraint functions at each iteration to show the performance of the algorithms, thus the code is very time-consuming.  Each pure time shown in the figures equals to the total time minus the time for computing  the true values of objective and constraint functions.



### Dependencies
- Python 3.6+
- Pytorch

### Supported algorithms
- BSGD
- SOAP
- MOAP
- SOX

### Scripts for reproducing the reported results

BSGD <=> SOX with gamma = 1.0, beta = 1.0; SOAP <=> SOX with beta = 1.0.

#### 1) CIFAR10 
```
python main_cifar10_resnet18.py
```
#### 2) CIFAR100
```

```
## How to cite
If you find our work useful, please consider citing [our paper](https://arxiv.org/pdf/2202.12396.pdf). 
```
@inproceedings{wang2022sox,
    title={Finite-Sum Coupled Compositional Stochastic Optimization: Theory and Applications},
    author={Wang, Bokun and Yang, Tianbao},
    journal={Proc. of the 39th International Conference on Machine Learning},
    year={2022}
}
```
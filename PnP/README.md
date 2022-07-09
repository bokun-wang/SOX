### Dependencies
- Python 3.6+
- Pytorch

### Supported algorithms

- BS-PnP
- BSGD
- SOAP
- MOAP
- SOX

### Scripts for reproducing the reported results

BSGD <=> SOX with gamma = 1.0, beta = 1.0; SOAP <=> SOX with beta = 1.0.

#### 1) ijcnn1
```
python main_bspnp.py --dataset ijcnn1
python main_sox.py --dataset ijcnn1 --lr 0.01 --gamma 1.0 --beta 1.0
python main_sox.py --dataset ijcnn1 --lr 0.1 --gamma 0.1 --beta 1.0
python main_sox.py --dataset ijcnn1 --lr 0.1 --gamma 0.1 --beta 0.9
python main_moap.py --dataset ijcnn1 --lr 0.005 --gamma 0.9 --beta 0.1
```
#### 2) covtype
```
python main_bspnp.py --dataset covtype
python main_sox.py --dataset covtype --lr 0.0005 --gamma 1.0 --beta 1.0
python main_sox.py --dataset covtype --lr 0.005 --gamma 0.5 --beta 1.0
python main_sox.py --dataset covtype --lr 0.001 --gamma 0.5 --beta 0.1
python main_moap.py --dataset covtype --lr 0.001 --gamma 0.5 --beta 0.1
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
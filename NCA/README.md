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

#### 1) MNIST Dataset
```
python main_sox.py --dataset mnist --lr 0.01 --gamma 1.0 --beta 1.0 
python main_sox.py --dataset mnist --lr 0.1 --gamma 0.9 --beta 1.0
python main_sox.py --dataset mnist --lr 0.1 --gamma 0.9 --beta 0.7 
python main_moap.py --dataset mnist --lr 0.01 --gamma 0.9 --beta 0.5
```
#### 2) Sensorless Dataset
```
python main_sox.py --dataset sensorless --lr 0.1 --gamma 1.0 --beta 1.0 
python main_sox.py --dataset sensorless --lr 2.0 --gamma 0.9 --beta 1.0
python main_sox.py --dataset sensorless --lr 2.0 --gamma 0.9 --beta 0.9 
python main_moap.py --dataset sensorless --lr 0.1 --gamma 0.9 --beta 0.7 
```
#### 3) USPS Dataset
```
python main_sox.py --dataset usps --lr 0.5 --gamma 1.0 --beta 1.0 
python main_sox.py --dataset usps --lr 0.5 --gamma 0.7 --beta 1.0
python main_sox.py --dataset usps --lr 0.5 --gamma 0.9 --beta 0.9
python main_moap.py --dataset usps --lr 0.5 --gamma 0.9 --beta 0.9 
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
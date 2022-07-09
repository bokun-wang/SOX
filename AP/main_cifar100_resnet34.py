from utils import resnet34
from torchvision import datasets
from config_cifar100 import return_conf
from train_eval import run_classification
from imbalanced_cifar import IMBALANCECIFAR100, transform_train, transform_val

# Fixed configs
seeds = [0, 1, 2]
loss_types = ['SOX', 'MOAPV2', 'SOAP', 'BSGD']
imb_ratio = 0.02
dataset = 'cifar100'
train_dataset = IMBALANCECIFAR100(root='./data', download=True, transform=transform_train, imb_factor=imb_ratio)
val_dataset = IMBALANCECIFAR100(root='./data', download=True, transform=transform_val, imb_factor=imb_ratio, val=True)
test_dataset = datasets.CIFAR100(root='./data', train=False, download=True, transform=transform_val)

if imb_ratio == 0.02:
    apdix = '002'
elif imb_ratio == 0.2:
    apdix = '020'
else:
    apdix = None

for ls_id, loss_type in enumerate(loss_types):
    for run, seed in enumerate(seeds):
        model = resnet34()
        model_name = 'resnet34'
        conf = return_conf(dataset=dataset, imb_ratio=imb_ratio, loss_type=loss_type)
        conf['model'] = model_name
        conf['pre_train'] = './cepretrainmodels/cifar100_' + model_name + '_' + apdix + '.ckpt'
        conf['imb_ratio'] = imb_ratio
        run_classification(train_dataset=train_dataset, val_dataset=val_dataset, test_dataset=test_dataset, model=model,
                           conf=conf, seed=seed)
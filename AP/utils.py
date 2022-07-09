from typing import Optional, Sized
from torchvision.models.resnet import ResNet, Bottleneck, BasicBlock
from torchvision.datasets.folder import ImageFolder
import numpy as np
from sklearn.metrics import auc, roc_auc_score, average_precision_score
from sklearn.metrics import precision_recall_curve
from torch.utils.data.sampler import Sampler
import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms
from PIL import Image

def set_all_seeds(SEED):
    # REPRODUCIBILITY
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class AUPRCSampler(Sampler):
    def __init__(self, labels, batchSize, posNum=1):
        # positive class: minority class
        # negative class: majority class

        self.labels = labels
        self.posNum = posNum
        self.batchSize = batchSize

        self.clsLabelList = np.unique(labels)
        self.dataDict = {}

        for label in self.clsLabelList:
            self.dataDict[str(label)] = []

        for i in range(len(self.labels)):
            self.dataDict[str(self.labels[i])].append(i)

        self.ret = []

    def __iter__(self):
        minority_data_list = self.dataDict[str(1)]
        majority_data_list = self.dataDict[str(0)]

        # print(len(minority_data_list), len(majority_data_list))
        np.random.shuffle(minority_data_list)
        np.random.shuffle(majority_data_list)

        # In every iteration : sample 1(posNum) positive sample(s), and sample batchSize - 1(posNum) negative samples
        if len(minority_data_list) // self.posNum > len(majority_data_list) // (
                self.batchSize - self.posNum):  # At this case, we go over the all positive samples in every epoch.
            # extend the length of majority_data_list from  len(majority_data_list) to len(minority_data_list)* (batchSize-posNum)
            majority_data_list.extend(np.random.choice(majority_data_list, len(minority_data_list) // self.posNum * (
                        self.batchSize - self.posNum) - len(majority_data_list), replace=True).tolist())

        elif len(minority_data_list) // self.posNum < len(majority_data_list) // (
                self.batchSize - self.posNum):  # At this case, we go over the all negative samples in every epoch.
            # extend the length of minority_data_list from len(minority_data_list) to len(majority_data_list)//(batchSize-posNum) + 1

            minority_data_list.extend(np.random.choice(minority_data_list, len(majority_data_list) // (
                        self.batchSize - self.posNum) * self.posNum - len(minority_data_list), replace=True).tolist())

        self.ret = []
        for i in range(len(minority_data_list) // self.posNum):
            self.ret.extend(minority_data_list[i * self.posNum:(i + 1) * self.posNum])
            startIndex = i * (self.batchSize - self.posNum)
            endIndex = (i + 1) * (self.batchSize - self.posNum)
            self.ret.extend(majority_data_list[startIndex:endIndex])

        return iter(self.ret)

    def __len__(self):
        return len(self.ret)


def resnet18():
    model = ResNet(BasicBlock, [2, 2, 2, 2], num_classes=1)
    return model


def resnet34():
    model = ResNet(BasicBlock, [3, 4, 6, 3], num_classes=1)
    return model


def resnet50():
    model = ResNet(Bottleneck, [3, 4, 6, 3], num_classes=1)
    return model


class MyImageFolder(ImageFolder):
    def __init__(self, root, transform):
        super().__init__(root, transform=transform)

    def __getitem__(self, index):
        sample, target = super().__getitem__(index)
        return index, sample, target


class TestImageDataset(Dataset):
    def __init__(self, images, targets, image_size=32):
        self.images = images.astype(np.uint8)
        self.targets = targets
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
            # transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = self.images[idx]
        target = self.targets[idx]
        image = Image.fromarray(image.astype('uint8'))
        image = self.transform(image)
        return image, target


def prc_auc(targets, preds):
    precision, recall, _ = precision_recall_curve(targets, preds)
    return auc(recall, precision)


def prc_recall_curve(targets, preds):
    precision, recall, _ = precision_recall_curve(targets, preds)


def ave_prc(targets, preds):
    return average_precision_score(targets, preds)


def compute_cla_metric(targets, preds, num_tasks):
    prc_results = []
    roc_results = []
    for i in range(num_tasks):
        is_labeled = targets[:, i] == targets[:, i]  ## filter some samples without groundtruth label
        target = targets[is_labeled, i]
        pred = preds[is_labeled, i]
        try:
            prc = prc_auc(target, pred)
        except ValueError:
            prc = np.nan
            print("In task #", i + 1, " , there is only one class present in the set. PRC is not defined in this case.")
        try:
            roc = roc_auc_score(target, pred)
        except ValueError:
            roc = np.nan
            print("In task #", i + 1, " , there is only one class present in the set. ROC is not defined in this case.")
        if not np.isnan(prc):
            prc_results.append(prc)
        else:
            print("PRC results do not consider task #", i + 1)
        if not np.isnan(roc):
            roc_results.append(roc)
        else:
            print("ROC results do not consider task #", i + 1)
    return prc_results, roc_results


def global_surrogate_loss_with_sqh(target, pred, threshold):
    posNum = np.sum(target)
    target, pred = target.reshape(-1), pred.reshape(-1)
    # print(target, pred)
    # print(posNum)
    loss = 0
    for t in range(len(target)):
        if target[t] == 1:
            # print(t)
            all_surr_loss = np.maximum(threshold - (pred[t] - pred), np.array([0] * len(target))) ** 2
            num = np.sum(all_surr_loss * (target == 1))
            dem = np.sum(all_surr_loss)

            loss += -num / dem

    return loss / posNum


def t_sigmoid(tensor, tau=1.0):
    # tau is the temperature.
    exponent = -tensor / tau
    y = 1.0 / (1.0 + torch.exp(exponent))
    return y


def t_exp(tensor, tau=1.0):
    # tau is the temperature.
    exponent = -tensor / tau
    y = torch.exp(exponent)
    return y

import numpy as np
import torch
import torch.nn as nn
import math
from utils import seed_everything
from torch.optim import SGD

class MOAP:
    def __init__(self, X_train, X_test, y_train, y_test, device, emb_dim=None,
                 init="random", gamma=0.9, beta=0.9, seed=None):
        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test
        self.num_train, self.input_dim = X_train.shape
        self.device = device
        self.emb_dim = emb_dim
        self.init = init
        self.gamma = gamma
        self.beta = beta
        self.seed = seed
        seed_everything(self.seed)

        # weight initialization
        if self.init == "identity":
            self.A = torch.eye(self.emb_dim, self.input_dim, device=self.device, requires_grad=True)
        elif self.init == "kaiming":
            self.A = torch.zeros(self.emb_dim, self.input_dim, device=self.device, requires_grad=True)
            nn.init.kaiming_uniform_(self.A, a=math.sqrt(5))
        else:
            raise ValueError("[!] {} initialization is not supported.".format(self.init))

    @staticmethod
    def _pairwise_l2_sq(x):
        """Compute pairwise squared Euclidean distances.
        """
        dot = torch.mm(x.double(), torch.t(x.double()))
        norm_sq = torch.diag(dot)
        dist = norm_sq[None, :] - 2 * dot + norm_sq[:, None]
        dist = torch.clamp(dist, min=0)  # replace negative values with 0
        return dist.float()

    @staticmethod
    def _softmax(x):
        """Compute row-wise softmax.

        Notes:
          Since the input to this softmax is the negative of the
          pairwise L2 distances, we don't need to do the classical
          numerical stability trick.
        """
        exp = torch.exp(x)
        return exp / exp.sum(dim=1)

    def train(self, num_iters, batch_size=None, lr=1e-4, weight_decay=10):
        seed_everything(self.seed)
        mom = 1.0 - self.beta
        self.optimizer = SGD([self.A], lr=lr, momentum=mom, weight_decay=weight_decay)
        self.u_all = torch.tensor([0.0] * self.num_train, requires_grad=False).to(self.device)
        self.u_pos = torch.tensor([0.0] * self.num_train, requires_grad=False).to(self.device)

        test_loss_trace = []

        for t in range(num_iters):
            index = np.random.choice(self.num_train, batch_size)
            index = torch.tensor(index).to(self.device)
            X_batch = self.X_train[index]
            y_batch = self.y_train[index]

            # zero out the gradient
            self.optimizer.zero_grad()

            # compute pairwise boolean class matrix
            y_mask = y_batch[:, None] == y_batch[None, :]
            y_mask.to(torch.float)

            diag_mask = 1.0 - torch.eye(y_mask.shape[0])

            embedding = torch.mm(X_batch, torch.t(self.A))

            distances = self._pairwise_l2_sq(embedding)

            all_loss = torch.exp(-distances) * diag_mask
            pos_loss = all_loss * y_mask

            all_loss = all_loss.sum(1, keepdim=False)
            pos_loss = pos_loss.sum(1, keepdim=False)

            denom = (1 - self.gamma) * self.u_all[index] + self.gamma * (self.num_train / batch_size) * all_loss
            numer = (1 - self.gamma) * self.u_pos[index] + self.gamma * (self.num_train / batch_size) * pos_loss
            loss = - torch.mean(numer / denom)

            # backward pass
            loss.backward()
            self.optimizer.step()

            self.u_all *= (1 - self.gamma)
            self.u_pos *= (1 - self.gamma)

            self.u_all[index] += self.gamma * all_loss.detach().clone()

            self.u_pos[index] += self.gamma * pos_loss.detach().clone()

            if t % 500 == 0:
                y_mask_test = self.y_test[:, None] == self.y_test[None, :]
                y_mask_test.to(torch.float)

                diag_mask_test = 1.0 - torch.eye(y_mask_test.shape[0])

                embedding_test = torch.mm(self.X_test, torch.t(self.A))

                distances_test = self._pairwise_l2_sq(embedding_test)

                all_loss_test = torch.exp(-distances_test) * diag_mask_test
                pos_loss_test = all_loss_test * y_mask_test

                all_loss_test = all_loss_test.sum(1, keepdim=False)
                pos_loss_test = pos_loss_test.sum(1, keepdim=False)

                loss = - torch.mean(pos_loss_test / all_loss_test)
                print(-loss.detach().clone().item())
                test_loss = loss.detach().clone().item()
                test_loss_trace.append(test_loss)

        return test_loss_trace

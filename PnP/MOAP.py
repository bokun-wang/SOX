import numpy as np
import torch
import torch.nn as nn
import math
from utils import seed_everything
from torch.optim import SGD
from sklearn.metrics import roc_auc_score
from my_timer import Timer


class MOAP:
    def __init__(self, X_train_ps, X_train_ns, X_test_ps, X_test_ns, device, gamma=0.9, beta=0.9,
                 seed=None):
        self.X_train_ps = X_train_ps
        self.X_train_ns = X_train_ns
        self.X_test_ps = X_test_ps
        self.X_test_ns = X_test_ns
        self.num_train_ps, self.input_dim = X_train_ps.shape
        self.num_train_ns = X_train_ns.shape[0]
        self.device = device
        self.gamma = gamma
        self.beta = beta
        self.seed = seed
        seed_everything(self.seed)
        self.timer = Timer()
        self.A = torch.zeros(self.input_dim, 1, device=self.device, requires_grad=True)
        nn.init.kaiming_uniform_(self.A, a=math.sqrt(5))

    @staticmethod
    def _pnp_fval(f_ps, f_ns, poly):
        f_ps = f_ps.view(-1)
        f_ns = f_ns.view(-1)
        mat_data = f_ns.repeat(len(f_ps), 1).T  # neg x pos
        f_ps = f_ps.view(-1, 1)
        inner_loss = torch.exp(-(f_ps.T - mat_data))
        loss = (inner_loss.mean(1)) ** poly
        return torch.mean(loss)

    def train(self, num_iters, batch_size_out=None, batch_size_in=None, lr=1e-4, weight_decay=10, poly=4):
        seed_everything(self.seed)
        mom = 1.0 - self.beta
        self.optimizer = SGD([self.A], lr=lr, momentum=mom, weight_decay=weight_decay)
        self.u_neg = torch.tensor([0.0] * self.num_train_ns).view(-1, 1).to(self.device)

        test_loss_trace = []
        test_auc_trace = []
        total_time = 0.0

        for t in range(num_iters):
            self.timer.start()
            index_ps = np.random.choice(self.num_train_ps, batch_size_in)
            index_ns = np.random.choice(self.num_train_ns, batch_size_out)
            index_ps = torch.tensor(index_ps).to(self.device)
            index_ns = torch.tensor(index_ns).to(self.device)

            X_batch_ps = self.X_train_ps[index_ps]
            X_batch_ns = self.X_train_ns[index_ns]

            # zero out the gradient
            self.optimizer.zero_grad()

            f_ps = torch.mm(X_batch_ps, self.A)
            f_ns = torch.mm(X_batch_ns, self.A)

            f_ps = f_ps.view(-1)
            f_ns = f_ns.view(-1)
            mat_data = f_ns.repeat(len(f_ps), 1).T  # neg x pos
            f_ps = f_ps.view(-1, 1)
            loss = torch.exp(-(f_ps.T - mat_data))
            self.u_neg *= (1 - self.gamma)
            self.u_neg[index_ns] += self.gamma * loss.mean(1)[:, None]
            p = poly * (self.u_neg[index_ns] ** (poly - 1))
            p.detach_()
            loss = torch.mean(p * loss)

            # backward pass
            loss.backward()
            self.optimizer.step()
            time_per_iter = self.timer.stop()
            total_time += time_per_iter

            if t % 500 == 0:
                f_ps_test = torch.mm(self.X_test_ps, self.A)
                f_ns_test = torch.mm(self.X_test_ns, self.A)
                test_loss = self._pnp_fval(f_ns=f_ns_test, f_ps=f_ps_test, poly=poly).item()
                f_test = np.concatenate((f_ps_test.detach().numpy(), f_ns_test.detach().numpy()), axis=0)
                y_test = np.concatenate(
                    (np.array([1.0] * self.X_test_ps.shape[0]), np.array([-1.0] * self.X_test_ns.shape[0])), axis=0)
                test_auc = roc_auc_score(y_test, f_test)
                print('Test loss: {}, Test AUC: {}'.format(test_loss, test_auc))
                test_auc_trace.append(test_auc)
                test_loss_trace.append(test_loss)

        return test_loss_trace, test_auc_trace, total_time
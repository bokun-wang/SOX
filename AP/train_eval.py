import os
import torch
import torch.nn.functional as F
from torch.optim import Adam, SGD
from utils import compute_cla_metric, ave_prc, global_surrogate_loss_with_sqh, AUPRCSampler, set_all_seeds
from torch.utils.data import DataLoader
import numpy as np
from losses import SOAPLOSS, MOAPV2LOSS, FocalLoss
from my_timer import Timer

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def run_classification(train_dataset, val_dataset, test_dataset, model, conf, seed):
    set_all_seeds(seed)
    if conf['dataset'] == "melanoma":
        n_train = 26500
        n_train_pos = 467
    else:
        if conf['imb_ratio'] == 0.02:
            n_train = 20400
            n_train_pos = 400
        elif conf['imb_ratio'] == 0.2:
            n_train = 24000
            n_train_pos = 4000
        else:
            raise ValueError('Imb factor not implemented!')

    model = model.to(device)
    if conf['pre_train'] is not None:
        print('we are loading pretrain model')
        state_key = torch.load(conf['pre_train'])
        if conf['dataset'] in ['cifar10', 'cifar100']:
            print('pretrain model is loaded from {} epoch'.format(state_key['epoch']))
        else:
            print('pretrain model is loaded')
        if conf['dataset'] in ['cifar10', 'cifar100']:
            filtered = {k: v for k, v in state_key['model'].items() if 'fc' not in k}
        else:
            filtered = {k: v for k, v in state_key.items() if 'fc' not in k}
        model.load_state_dict(filtered, False)
    if conf['ft_mode'] == 'frozen':
        for key, param in model.named_parameters():
            if 'fc' in key and 'gn' not in key:
                param.requires_grad = True
            else:
                param.requires_grad = False
    elif conf['ft_mode'] == 'fc_random':
        model.fc.reset_parameters()

    if conf['loss_type'] in ['MOAPV2', 'SOX']:
        optimizer = SGD(model.parameters(), lr=conf['lr'], momentum=conf['grad_mom'], weight_decay=conf['weight_decay'])
    else:
        optimizer = SGD(model.parameters(), lr=conf['lr'], weight_decay=conf['weight_decay'])

    global u, a, b, m, alpha
    labels = [0] * (n_train - n_train_pos) + [1] * n_train_pos
    if conf['loss_type'] in ['SOAP', 'SOX']:
        # Don't add the length of validation set
        criterion = SOAPLOSS(threshold=conf['threshold'], data_length=len(train_dataset))
    elif conf['loss_type'] == 'MOAPV2':
        criterion = MOAPV2LOSS(threshold=conf['threshold'], data_length=len(train_dataset),
                               n_pos_total=n_train_pos)
    elif conf['loss_type'] == 'CE':
        criterion = torch.nn.BCEWithLogitsLoss(reduction='none')
    elif conf['loss_type'] == 'FocalLoss':
        n_pos = n_train_pos
        n_neg = n_train - n_train_pos
        cls_num_list = [n_neg, n_pos]
        criterion = FocalLoss(cls_num_list=cls_num_list)
    else:
        raise ValueError

    val_loader = DataLoader(val_dataset, conf['vt_batch_size'], shuffle=False, num_workers=16, pin_memory=True)
    test_loader = DataLoader(test_dataset, conf['vt_batch_size'], shuffle=False, num_workers=16, pin_memory=True)

    best_auprc_score = 0
    final_auprc = 0
    best_test_auprc_score = 0

    epochs_trace = []
    iters_trace = []
    val_AP_trace = []
    tr_AP_trace = []
    te_AP_trace = []
    tr_loss_trace = []
    time_pb_trace = []  # time per batch

    timer = Timer()

    for epoch in range(1, conf['epochs'] + 1):
        if conf['loss_type'] in ['CE', 'FocalLoss']:
            train_loader = DataLoader(train_dataset, conf['batch_size'], shuffle=True, drop_last=True, num_workers=16,
                                      pin_memory=True)
        else:
            train_loader = DataLoader(train_dataset, batch_size=conf['batch_size'],
                                      sampler=AUPRCSampler(labels, conf['batch_size'], posNum=conf['posNum']),
                                      num_workers=16,
                                      pin_memory=True)
        timer.start()
        avg_train_loss = train_classification(model, optimizer, train_loader, criterion=criterion, conf=conf,
                                              epoch=epoch)
        time_train_epoch = timer.stop()
        time_per_batch = time_train_epoch / len(train_loader)
        if epoch % 2 == 0:
            train_auprc, train_roc, train_ap, train_surr_loss = val_train_classification(model, train_loader, conf)
            if conf['dataset'] in ['cifar10', 'cifar100']:
                val_auprc, val_roc, val_ap, val_surr_loss = val_train_classification(model, val_loader, conf)
            else:
                val_auprc, val_roc, val_ap, val_surr_loss = test_classification(model, val_loader, conf)
            test_auprc, test_roc, test_ap, test_surr_loss = test_classification(model, test_loader, conf)
            if best_test_auprc_score <= np.mean(test_auprc):
                best_test_auprc_score = np.mean(test_auprc)
            print('Train AP {:.4f}, Val AP: {}, Test AP: {:.4f}\n'.format(train_ap, val_ap, test_ap))
            # record the traces
            epochs_trace.append(epoch)
            iters_trace.append(epoch * len(train_loader))
            val_AP_trace.append(val_ap)
            tr_loss_trace.append(train_surr_loss)
            tr_AP_trace.append(train_ap)
            te_AP_trace.append(test_ap)
            time_pb_trace.append(time_per_batch)

        if epoch in conf['epochs_decay']:
            for param_group in optimizer.param_groups:
                param_group['lr'] = param_group['lr'] / conf['lr_decay_factor']


    epochs_trace = np.array(epochs_trace)
    iters_trace = np.array(iters_trace)
    val_AP_trace = np.array(val_AP_trace)
    tr_loss_trace = np.array(tr_loss_trace)
    tr_AP_trace = np.array(tr_AP_trace)
    te_AP_trace = np.array(te_AP_trace)
    time_pb_trace = np.array(time_pb_trace)

    if conf['imb_ratio'] == 0.02:
        apdix = '002'
    elif conf['imb_ratio'] == 0.2:
        apdix = '020'
    else:
        apdix = '000'

    if conf['loss_type'] in ['CE', 'FocalLoss']:
        file_name = "./results/{}-{}-{}-{}-{}-{}-{}_traces.npz".format(str(seed), conf['dataset'], apdix,
                                                                    conf['loss_type'],
                                                                    conf['lr'], conf['model'], conf['epochs'])
    elif conf['loss_type'] == 'SOAP':
        file_name = "./results/{}-{}-{}-{}-{}-{}-{}-{}-{}_traces.npz".format(str(seed), conf['dataset'], apdix,
                                                                          conf['loss_type'],
                                                                          conf['lr'], conf['mv_gamma'], conf['posNum'],
                                                                          conf['model'], conf['epochs'])
    elif conf['loss_type'] == 'MOAPV2':
        file_name = "./results/{}-{}-{}-{}-{}-{}-{}-{}-{}-{}_traces.npz".format(str(seed), conf['dataset'], apdix,
                                                                                conf['loss_type'], conf['lr'],
                                                                                conf['mv_gamma'], conf['grad_mom'],
                                                                                conf['posNum'], conf['model'],
                                                                                conf['epochs'])
    elif conf['loss_type'] == 'SOX':
        file_name = "./results/{}-{}-{}-{}-{}-{}-{}-{}-{}-{}-{}_traces.npz".format(str(seed), conf['dataset'], apdix,
                                                                                   conf['loss_type'], conf['lr'],
                                                                                   conf['mv_gamma'], conf['grad_mom'],
                                                                                   conf['posNum'], conf['batch_size'],
                                                                                   conf['model'], conf['epochs'])
    else:
        file_name = None
    np.savez(file_name, epochs_trace=epochs_trace, iters_trace=iters_trace, val_AP_trace=val_AP_trace,
             tr_AP_trace=tr_AP_trace, te_AP_trace=te_AP_trace, tr_loss_trace=tr_loss_trace, time_pb_trace=time_pb_trace)


def train_classification(model, optimizer, train_loader, criterion, conf, epoch):
    model.train()
    global a, b, m, alpha
    losses = []
    for i, (index, inputs, target) in enumerate(train_loader):
        if i % 50 == 0:
            print(epoch, " : ", i, "/", len(train_loader))
        optimizer.zero_grad()
        inputs = inputs.to(device)
        target = target.to(device).float()
        out = model(inputs)
        if conf['loss_type'] == 'CE':
            if len(target.shape) != 2:
                target = torch.reshape(target, (-1, conf['num_tasks']))
            loss = criterion(out, target)
            loss = loss.sum()
            loss.backward()
            optimizer.step()
        elif conf['loss_type'] in ['SOAP', 'SOX', 'MOAPV2']:
            predScore = torch.nn.Sigmoid()(out)
            loss = criterion(f_ps=predScore[0:conf['posNum']], f_ns=predScore[conf['posNum']:],
                             index_s=index[0:conf['posNum']], gamma=conf['mv_gamma'])
            loss.backward()
            optimizer.step()
        elif conf['loss_type'] == 'FocalLoss':
            loss = criterion(out, target, epoch)
            loss.backward()
            optimizer.step()

        losses.append(loss)
    return sum(losses).item() / len(losses)


def val_train_classification(model, test_loader, conf):
    model.eval()
    preds = torch.Tensor([]).to(device)
    targets = torch.Tensor([]).to(device)
    for (index, inputs, target) in test_loader:

        inputs = inputs.to(device)
        target = target.to(device).float()
        with torch.no_grad():
            out = model(inputs)
        if len(target.shape) != 2:
            target = torch.reshape(target, (-1, conf['num_tasks']))
        if out.shape[1] == 1:
            pred = torch.sigmoid(out)  ### prediction real number between (0,1)
        else:
            pred = torch.softmax(out, dim=-1)[:, 1:2]
        preds = torch.cat([preds, pred], dim=0)
        targets = torch.cat([targets, target], dim=0)

    auprc, auroc = compute_cla_metric(targets.cpu().detach().numpy(), preds.cpu().detach().numpy(), conf['num_tasks'])
    ap = ave_prc(targets.cpu().detach().numpy(), preds.cpu().detach().numpy())

    surro_loss = global_surrogate_loss_with_sqh(targets.cpu().detach().numpy(), preds.cpu().detach().numpy(),
                                                conf['threshold'])

    return auprc, auroc, ap, surro_loss


def test_classification(model, test_loader, conf):
    model.eval()
    preds = torch.Tensor([]).to(device)
    targets = torch.Tensor([]).to(device)

    for (inputs, target) in test_loader:
        inputs = inputs.to(device)
        target = target.to(device).float()
        if conf['dataset'] == 'cifar10':
            target[target <= 4] = 0
            target[target > 4] = 1
        elif conf['dataset'] == 'cifar100':
            target[target <= 49] = 0
            target[target > 49] = 1

        with torch.no_grad():
            out = model(inputs)
        if len(target.shape) != 2:
            target = torch.reshape(target, (-1, conf['num_tasks']))

        if out.shape[1] == 1:
            pred = torch.sigmoid(out)  ### prediction real number between (0,1)
        else:
            pred = torch.softmax(out, dim=-1)[:, 1:2]
        preds = torch.cat([preds, pred], dim=0)
        targets = torch.cat([targets, target], dim=0)
    auprc, auroc = compute_cla_metric(targets.cpu().detach().numpy(), preds.cpu().detach().numpy(), conf['num_tasks'])
    ap = ave_prc(targets.cpu().detach().numpy(), preds.cpu().detach().numpy())

    surro_loss = global_surrogate_loss_with_sqh(targets.cpu().detach().numpy(), preds.cpu().detach().numpy(),
                                                conf['threshold'])

    return auprc, auroc, ap, surro_loss

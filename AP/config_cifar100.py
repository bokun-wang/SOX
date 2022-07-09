def return_conf(dataset, imb_ratio, loss_type):
    # Common configs
    conf = {}
    conf['dataset'] = dataset
    conf['imb_ratio'] = imb_ratio
    conf['input_size'] = 384
    conf['num_tasks'] = 1
    conf['dropout'] = 0
    conf['weight_decay'] = 2e-4
    conf['epochs'] = 200
    conf['threshold'] = 0.6
    conf['lr_decay_factor'] = 10
    conf['epochs_decay'] = [100, 150]
    conf['dropout'] = 0
    conf['ft_mode'] = 'fc_random'
    conf['batch_size'] = 64
    conf['vt_batch_size'] = 64

    if loss_type == 'CE':  # configs for CE loss
        conf['lr'] = 1e-4
        conf['loss_type'] = 'CE'
    elif loss_type == 'BSGD':  # configs for BSGD
        conf['lr'] = 0.005
        conf['loss_type'] = 'SOAP'
        conf['mv_gamma'] = 1.0
        conf['posNum'] = conf['batch_size'] // 2
    elif loss_type == 'SOAP':  # configs for SOAP
        conf['lr'] = 0.005
        conf['loss_type'] = 'SOAP'
        conf['mv_gamma'] = 0.9
        conf['posNum'] = conf['batch_size'] // 2
    elif loss_type == 'SOX':  # configs for SOX
        conf['lr'] = 0.001
        conf['loss_type'] = 'SOX'
        conf['mv_gamma'] = 0.7
        conf['posNum'] = conf['batch_size'] // 2
        conf['grad_mom'] = 0.9
    elif loss_type == 'MOAPV2':  # configs for MOAPV2
        conf['lr'] = 0.005
        conf['loss_type'] = 'MOAPV2'
        conf['mv_gamma'] = 0.5
        conf['posNum'] = conf['batch_size'] // 2
        conf['grad_mom'] = 0.9
    else:
        raise ValueError('Unknown loss type')

    return conf

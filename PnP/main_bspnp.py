import argparse
import numpy as np
from my_timer import Timer
from scipy.optimize import fsolve
from sklearn.datasets import load_svmlight_file
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split


def main(args):

    # load the dataset
    if args.dataset == 'ijcnn1':
        X, y = load_svmlight_file('./data/ijcnn1.tr.bz2')
        X = X.toarray()
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.1, random_state=42
        )
        scaler = MinMaxScaler(feature_range=(0, 1))
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)
        if np.all(np.unique(y_train) == [-1, 1]) == False:
            raise ValueError
    elif args.dataset == 'covtype':
        X_full, y_full = load_svmlight_file('./data/covtype.libsvm.binary.scale.bz2')
        index_1 = np.where(y_full == 1)[0]
        index_2 = np.where(y_full == 2)[0]
        selected_index_1 = index_1[:len(index_1) // 10]
        index = np.concatenate((selected_index_1, index_2), axis=0)
        X = X_full[index]
        y = y_full[index]
        y[y == 1] = 1
        y[y == 2] = -1
        X = X.toarray()
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, train_size=100000, test_size=10000, random_state=42
        )
        scaler = MinMaxScaler(feature_range=(0, 1))
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)
        if np.all(np.unique(y_train) == [-1, 1]) == False:
            raise ValueError
    else:
        raise ValueError('Unknown dataset')

    # flip the label
    index_ps_tr = np.where(y_train == -1)[0]
    index_ns_tr = np.where(y_train == 1)[0]
    index_ps_te = np.where(y_test == -1)[0]
    index_ns_te = np.where(y_test == 1)[0]

    Xtrain_pos = X_train[index_ps_tr]
    Xtrain_neg = X_train[index_ns_tr]
    Xtest_pos = X_test[index_ps_te]
    Xtest_neg = X_test[index_ns_te]

    Ytrain_pos = np.ones_like(Xtrain_pos)
    Ytrain_neg = - np.ones_like(Xtrain_neg)
    n_pos, n_neg = len(Ytrain_pos), len(Ytrain_neg)
    p = 4
    dim = Xtrain_pos.shape[1]

    # bs_pnp
    pnp_timer = Timer()
    dik = np.ones((n_neg, n_pos)) / (n_pos * n_neg)
    lam = np.zeros(dim)
    T_max = args.num_iters
    n_lrns = dim  # use features as weak rankers
    total_time_pnp = 0.0
    for t in range(T_max):
        pnp_timer.start()
        max_score = -100000.0
        jmax = 0
        for j in range(n_lrns):
            sum_dik = np.power(np.sum(dik, axis=1), p - 1)  # size: [n_neg,]
            pred_pos = np.squeeze(Xtrain_pos[:, j])  # size: [n_pos,]
            pred_neg = np.squeeze(Xtrain_neg[:, j])  # size: [n_neg,]
            # weighted_dik = np.array([np.dot(np.squeeze(dik[k,:]), pred_pos - pred_neg[k]) for k in range(n_neg)])
            weighted_dik = np.squeeze(np.sum(dik * (pred_pos - pred_neg[:, None]), axis=1))
            new_score = np.dot(sum_dik, weighted_dik)
            if new_score >= max_score:
                max_score = new_score
                jmax = j

        # root finding
        pred_pos_max = np.squeeze(Xtrain_pos[:, jmax])  # size: [n_pos,]
        pred_neg_max = np.squeeze(Xtrain_neg[:, jmax])
        func = lambda alpha: np.sum(
            np.power(np.squeeze(np.sum(dik * np.exp(-alpha * (pred_pos_max - pred_neg_max[:, None])), axis=1)),
                     p - 1) * np.squeeze(np.sum(
                (pred_pos_max - pred_neg_max[:, None]) * dik * np.exp(-alpha * (pred_pos_max - pred_neg_max[:, None])),
                axis=1)))
        alpha_t = fsolve(func, x0=np.array(0.))

        # coordinate update
        lam[jmax] += alpha_t

        # update dik
        z_t = np.sum(np.sum(dik * np.exp(-alpha_t * (pred_pos_max - pred_neg_max[:, None]))))
        dik = dik * np.exp(-alpha_t * (pred_pos_max - pred_neg_max[:, None])) / z_t

        time_per_iter_pnp = pnp_timer.stop()
        total_time_pnp += time_per_iter_pnp

    f_ps = np.dot(Xtest_pos, lam)[:, None]
    f_ns = np.dot(Xtest_neg, lam)[:, None]
    mat_data = f_ns.repeat(len(f_ps), 1)  # neg x pos
    inner_loss = np.exp(-(f_ps.T - mat_data))
    loss = (inner_loss.mean(1)) ** p
    test_loss_pnp = np.mean(loss)

    print("BS-PNP: Time: {0}, Test loss: {1}".format(total_time_pnp, test_loss_pnp))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="covtype", help="dataset: covtype or ijcnn1.")
    parser.add_argument("--num_iters", type=int, default=10, help="Number of iterations.")
    args, unparsed = parser.parse_known_args()
    main(args)

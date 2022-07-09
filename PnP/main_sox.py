import argparse
import numpy as np
import torch

from SOX import SOX
from sklearn.datasets import load_svmlight_file
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split


def main(args):
    use_cuda = args.cuda and torch.cuda.is_available()
    if use_cuda:
        device = torch.device("cuda")
    else:
        print("[*] Using cpu.")
        device = torch.device("cpu")

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

    X_train_ps = X_train[index_ps_tr]
    X_train_ns = X_train[index_ns_tr]
    X_test_ps = X_test[index_ps_te]
    X_test_ns = X_test[index_ns_te]

    X_train_ps = torch.tensor(X_train_ps, device=device, requires_grad=False).float()
    X_train_ns = torch.tensor(X_train_ns, device=device, requires_grad=False).float()
    X_test_ps = torch.tensor(X_test_ps, device=device, requires_grad=False).float()
    X_test_ns = torch.tensor(X_test_ns, device=device, requires_grad=False).float()

    all_loss_traces = []
    all_auc_traces = []
    all_total_time = []
    all_final_loss = []
    for seed in range(args.num_seeds):
        print("===========seed: {}===========".format(seed))
        sox = SOX(X_train_ps=X_train_ps, X_train_ns=X_train_ns, X_test_ps=X_test_ps, X_test_ns=X_test_ns, device=device,
                  gamma=args.gamma, beta=args.beta, seed=seed)
        test_loss_trace, test_auc_trace, total_time = sox.train(num_iters=args.num_iters,
                                                              batch_size_out=args.batch_size_out,
                                                              batch_size_in=args.batch_size_in, lr=args.lr,
                                                              weight_decay=args.wd)
        all_loss_traces.append(test_loss_trace)
        all_auc_traces.append(test_auc_trace)
        all_total_time.append(total_time)
        all_final_loss.append(test_loss_trace[-1])

    print("SOX: avg time: {:.5f}, std: {:.5f}, avg te loss {:.5f}, std: {:.5f}".format(np.average(all_total_time),
                                                                                       np.std(all_total_time),
                                                                                       np.average(all_final_loss),
                                                                                       np.std(all_final_loss)))
    if args.gamma == 1.0 and args.beta == 1.0:
        file_name = "./results/BSGD-{}_traces.npz".format(args.dataset)
    elif args.gamma < 1.0 and args.beta == 1.0:
        file_name = "./results/SOAP-{}_traces.npz".format(args.dataset)
    else:
        file_name = "./results/SOX-{}-{}-{}-{}-{}_traces.npz".format(args.gamma, args.beta, args.batch_size_in,
                                                                     args.batch_size_out, args.dataset)

    np.savez(file_name, all_loss_traces=all_loss_traces, all_auc_traces=all_auc_traces)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_seeds", type=int, default=5, help="Number of seeds.")
    parser.add_argument("--dataset", type=str, default="covtype", help="dataset.")
    parser.add_argument("--num_iters", type=int, default=10000, help="Number of iterations.")
    parser.add_argument("--batch_size_out", type=int, default=32, help="Batch size out.")
    parser.add_argument("--batch_size_in", type=int, default=32, help="Batch size in.")
    parser.add_argument("--lr", type=float, default=0.001, help="Learning rate.")
    parser.add_argument("--gamma", type=float, default=0.5, help="Avg.")
    parser.add_argument("--beta", type=float, default=0.1, help="Mom.")
    parser.add_argument("--wd", type=float, default=1e-3, help="Weight decay.")
    parser.add_argument("--cuda", type=lambda x: x.lower() in ['true', '1'], default=False,
                        help="Whether to use the GPU.")
    args, unparsed = parser.parse_known_args()
    main(args)

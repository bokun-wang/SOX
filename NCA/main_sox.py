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
    if args.dataset == 'mnist':
        X, y = load_svmlight_file('./data/mnist.scale.bz2')
    elif args.dataset == 'sensorless':
        X, y = load_svmlight_file('./data/Sensorless.scale')
    elif args.dataset == 'usps':
        X, y = load_svmlight_file('./data/usps.bz2')
    else:
        raise ValueError('Unknown dataset')
    X = X.toarray()
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.1, random_state=42
    )
    scaler = MinMaxScaler(feature_range=(0,1))
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    X_train = torch.tensor(X_train, device=device, requires_grad=False).float()
    y_train = torch.tensor(y_train, device=device, requires_grad=False).float()
    X_test= torch.tensor(X_test, device=device, requires_grad=False).float()
    y_test = torch.tensor(y_test, device=device, requires_grad=False).float()

    all_loss_traces = []
    for seed in range(args.num_seeds):
        print("===========seed: {}===========".format(seed))
        # SOX + kNN
        if args.dataset == 'mnist':
            emb_dim = 32
        elif args.dataset == 'sensorless':
            emb_dim = 4
        elif args.dataset == 'usps':
            emb_dim = 10
        else:
            raise ValueError('Unknown dataset')
        sox = SOX(X_train=X_train, X_test=X_test, y_train=y_train, y_test=y_test, device=device,
                  emb_dim=emb_dim, init=args.init, gamma=args.gamma, beta=args.beta, seed=seed)
        test_loss_trace = sox.train(num_iters=args.num_iters, batch_size=args.batch_size, lr=args.lr,
                                    weight_decay=args.wd)
        all_loss_traces.append(test_loss_trace)
    if args.gamma == 1.0 and args.beta == 1.0:
        file_name = "./results/BSGD-{}_traces.npz".format(args.dataset)
    elif args.gamma < 1.0 and args.beta == 1.0:
        file_name = "./results/SOAP-{}_traces.npz".format(args.dataset)
    else:
        file_name = "./results/SOX-{}_traces.npz".format(args.dataset)

    np.savez(file_name, all_loss_traces=all_loss_traces)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_seeds", type=int, default=5, help="Number of seeds.")
    parser.add_argument("--init", type=str, default="kaiming", help="Which initialization to use.")
    parser.add_argument("--dataset", type=str, default="mnist", help="dataset.")
    parser.add_argument("--num_iters", type=int, default=10000, help="Number of iterations.")
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size.")
    parser.add_argument("--lr", type=float, default=0.1, help="Learning rate.")
    parser.add_argument("--gamma", type=float, default=1.0, help="Avg.")
    parser.add_argument("--beta", type=float, default=1.0, help="Mom.")
    parser.add_argument("--wd", type=float, default=1e-3, help="Weight decay.")
    parser.add_argument("--cuda", type=lambda x: x.lower() in ['true', '1'], default=False,
                        help="Whether to use the GPU.")
    args, unparsed = parser.parse_known_args()
    main(args)

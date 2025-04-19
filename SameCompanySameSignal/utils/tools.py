import os
import itertools
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

from sklearn.linear_model import RidgeClassifier
import pandas as pd
import math
from sklearn.metrics import pairwise
from sklearn.metrics.pairwise import KERNEL_PARAMS
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.metrics import mean_squared_error
#
# from cvxopt import matrix, solvers
# solvers.options['show_progress'] = False
plt.switch_backend('agg')
from sklearn.base import BaseEstimator, RegressorMixin
import numpy as np


class TSMixerWrapper(BaseEstimator, RegressorMixin):
    def __init__(self, model, epochs=10, batch_size=16, lr=0.001):
        self.model = model
        self.epochs = epochs
        self.batch_size = batch_size
        self.lr = lr
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
        self.criterion = nn.MSELoss()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)

    def fit(self, X, y):
        self.model.train()
        # X, y = torch.tensor(X, dtype=torch.float32).to(self.device), torch.tensor(y, dtype=torch.float32).to(
        #     self.device)
        # dataset = torch.utils.data.TensorDataset(X, y)
        # dataloader = torch.utils.data.DataLoader(dataset, batch_size=self.batch_size, shuffle=True)
        for epoch in range(self.epochs):
            self.optimizer.zero_grad()
            X = torch.tensor(X).to(self.device)
            y = torch.tensor(y).to(self.device)
            outputs = self.model(X, None, None, None)
            loss = self.criterion(outputs, y)
            loss.backward()
            self.optimizer.step()
        return self

    def predict(self, X):
        self.model.eval()
        with torch.no_grad():
            X = torch.tensor(X).to(self.device)
            outputs = self.model(X, None, None, None)
        return outputs.cpu().numpy().squeeze()


def gaussian_kernel(x, y, sigma=1.0):
    return np.exp(-np.linalg.norm(x - y)**2 / (2 * (sigma ** 2)))


def compute_kernel_matrix(X, Y=None, sigma=1.0):
    n = X.shape[0]
    m = Y.shape[0] if Y is not None else n
    K = np.zeros((n, m))
    for i in range(n):
        for j in range(m):
            K[i, j] = gaussian_kernel(X[i], Y[j] if Y is not None else X[j], sigma)
    return K


# def call_kmm(source, target, B=10, sigma=1.0):
#     n_S = source.shape[0]
#     n_T = target.shape[0]
#
#     K = compute_kernel_matrix(source, sigma=sigma)
#     K_T = compute_kernel_matrix(source, target, sigma=sigma)
#
#     K = matrix(K)
#     K_T = matrix(K_T)
#
#     P = K
#     q = -np.mean(K_T, axis=1)
#     q = matrix(q)
#
#     G = matrix(np.vstack((np.eye(n_S), -np.eye(n_S))))
#     h = matrix(np.hstack((np.ones(n_S) * B, np.zeros(n_S))))
#
#     A = matrix(1.0, (1, n_S))
#     b = matrix(float(n_S))
#
#     sol = solvers.qp(P, q, G, h, A, b)
#     weights = np.array(sol['x'])
#     return weights


EPS = np.finfo(float).eps

class KLIEP:
    def __init__(self, estimator='Ridge', Xt=None, kernel="rbf", sigmas=None, max_centers=100, cv=3, algo="FW",
                 lr=[0.001, 0.01, 0.1, 1.0, 10.0], tol=1e-6, max_iter=2000, copy=True, verbose=1, random_state=None,
                 **params):
        if estimator == 'Ridge':
            self.estimator = RidgeClassifier()
        self.Xt = Xt
        self.kernel = kernel
        self.sigmas = sigmas
        self.max_centers = max_centers
        self.cv = cv
        self.algo = algo
        self.lr = lr
        self.tol = tol
        self.max_iter = max_iter
        self.copy = copy
        self.verbose = verbose
        self.random_state = random_state
        self.params = params

    def fit_weights(self, Xs, Xt, **kwargs):

        self.j_scores_ = {}

        # LCV GridSearch
        kernel_params = {k: v for k, v in self.__dict__.items() if k in KERNEL_PARAMS[self.kernel]}

        # Handle deprecated sigmas (will be removed)
        if (self.sigmas is not None) and ("gamma" not in kernel_params):
            kernel_params["gamma"] = self.sigmas

        params_dict = {k: (v if hasattr(v, "__iter__") else [v]) for k, v in kernel_params.items()}
        options = params_dict
        keys = options.keys()
        values = (options[key] for key in keys)
        params_comb = [dict(zip(keys, combination)) for combination in itertools.product(*values)]

        if len(params_comb) > 1:
            # Cross-validation process
            if len(Xt) < self.cv:
                raise ValueError("Length of Xt is smaller than cv value")

            if self.verbose:
                print("Cross Validation process...")

            shuffled_index = np.arange(len(Xt))
            np.random.shuffle(shuffled_index)

            max_ = -np.inf
            for params in params_comb:
                cv_scores = self._cross_val_jscore(Xs, Xt[shuffled_index], params, self.cv)
                self.j_scores_[str(params)] = np.mean(cv_scores)

                if self.verbose:
                    print("Parameters %s -- J-score = %.3f (%.3f)" %
                          (str(params), np.mean(cv_scores), np.std(cv_scores)))

                if self.j_scores_[str(params)] > max_:
                    self.best_params_ = params
                    max_ = self.j_scores_[str(params)]
        else:
            self.best_params_ = params_comb[0]

        self.alphas_, self.centers_ = self._fit(Xs, Xt, self.best_params_)

        self.weights_ = np.dot(
            pairwise.pairwise_kernels(Xs, self.centers_, metric=self.kernel, **self.best_params_),
            self.alphas_
        ).ravel()
        return self.weights_

    def _fit(self, Xs, Xt, kernel_params):
        if self.algo == "original":
            return self._fit_PG(Xs, Xt, PG=False, kernel_params=kernel_params)
        elif self.algo == "PG":
            return self._fit_PG(Xs, Xt, PG=True, kernel_params=kernel_params)
        elif self.algo == "FW":
            return self._fit_FW(Xs, Xt, kernel_params=kernel_params)
        else:
            raise ValueError("%s is not a valid value of algo" % self.algo)

    def _fit_FW(self, Xs, Xt, kernel_params):
      centers, A, b = self._centers_selection(Xs, Xt, kernel_params)

      alpha = 1 / (len(centers) * b)
      alpha = alpha.reshape(-1, 1)
      objective = np.sum(np.log(np.dot(A, alpha) + EPS))
      if self.verbose > 1:
          print("Alpha's optimization : iter %i -- Obj %.4f" % (0, objective))
      k = 0
      while k < self.max_iter:
          previous_objective = objective
          alpha_p = np.copy(alpha)
          r = 1. / np.clip(np.dot(A, alpha), EPS, np.inf)
          g = np.dot(np.transpose(A), r)
          B = np.diag(1 / b.ravel())
          LP = np.dot(g.transpose(), B)
          lr = 2 / (k + 2)
          alpha = (1 - lr) * alpha + lr * B[np.argmax(LP)].reshape(-1, 1)
          objective = np.sum(np.log(np.dot(A, alpha) + EPS))
          k += 1

          if self.verbose > 1:
              if k % 100 == 0:
                  print("Alpha's optimization : iter %i -- Obj %.4f" % (k, objective))
      return alpha, centers

    def _centers_selection(self, Xs, Xt, kernel_params):
        A = np.empty((Xt.shape[0], 0))
        b = np.empty((0,))
        centers = np.empty((0, Xt.shape[1]))

        max_centers = min(len(Xt), self.max_centers)
        np.random.seed(self.random_state)
        index = np.random.permutation(Xt.shape[0])

        k = 0

        while k * max_centers < len(index) and len(centers) < max_centers and k < 3:
            index_ = index[k * max_centers:(k + 1) * max_centers]
            centers_ = Xt[index_]
            A_ = pairwise.pairwise_kernels(Xt, centers_, metric=self.kernel, **kernel_params)
            B_ = pairwise.pairwise_kernels(centers_, Xs, metric=self.kernel, **kernel_params)
            b_ = np.mean(B_, axis=1)
            mask = (b_ < EPS).ravel()
            if np.sum(~mask) > 0:
                centers_ = centers_[~mask]
                centers = np.concatenate((centers, centers_), axis=0)
                A = np.concatenate((A, A_[:, ~mask]), axis=1)
                b = np.append(b, b_[~mask])
            k += 1

        if len(centers) >= max_centers:
            centers = centers[:max_centers]
            A = A[:, :max_centers]
            b = b[:max_centers]
        elif len(centers) > 0:
            print("Not enough centers, only %i centers found. Maybe consider a different value of kernel parameter." % len(centers))
        else:
            raise ValueError("No centers found! Please change the value of kernel parameter.")

        return centers, A, b.reshape(-1, 1)


def adjust_learning_rate(optimizer, epoch, args, verbose=True):
    # lr = args.learning_rate * (0.2 ** (epoch // 2))
    if args.lradj == 'type1':
        lr_adjust = {epoch: args.learning_rate * (0.5 ** ((epoch - 1) // 1))}
    elif args.lradj == 'type2':
        lr_adjust = {
            2: 5e-5, 4: 1e-5, 6: 5e-6, 8: 1e-6,
            10: 5e-7, 15: 1e-7, 20: 5e-8
        }
    elif args.lradj == "cosine":
        lr_adjust = {epoch: args.learning_rate / 2 * (1 + math.cos(epoch / args.train_epochs * math.pi))}
    if epoch in lr_adjust.keys():
        lr = lr_adjust[epoch]
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
        if verbose:
            print('Updating learning rate to {}'.format(lr))


def print_and_save(output_path, model, iof_dir, info):
    # Append info to the file, creating the file if it does not exist
    if not output_path.endswith('txt'):
        output_path = output_path + ('.txt')

    folder_path = './earnings_model_logs/' + model + '/' + iof_dir + '/'
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)

    path = os.path.join(folder_path, output_path)
    with open(path, 'a+') as file:  # 'a+' mode opens the file for both appending and reading
        file.write(info + '\n')  # Add the info with a newline

    print(info)


class EarlyStopping:
    def __init__(self, patience=7, verbose=False, delta=1e-4):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta

    def __call__(self, val_loss, model, path):
        score = -val_loss
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model, path)
        elif score < self.best_score + self.delta:
            self.counter += 1
            if self.verbose:
                print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model, path)
            self.counter = 0

    def save_checkpoint(self, val_loss, model, path):
        if self.verbose:
            print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        torch.save(model.state_dict(), path + '/' + 'checkpoint.pth')
        self.val_loss_min = val_loss


class dotdict(dict):
    """dot.notation access to dictionary attributes"""
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__


class StandardScaler():
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def transform(self, data):
        return (data - self.mean) / self.std

    def inverse_transform(self, data):
        return (data * self.std) + self.mean


def visual(true, preds=None, name='./pic/test.pdf'):
    """
    Results visualization
    """
    plt.figure()
    plt.plot(true, label='GroundTruth', linewidth=2)
    if preds is not None:
        plt.plot(preds, label='Prediction', linewidth=2)
    plt.legend()
    plt.savefig(name, bbox_inches='tight')


def adjustment(gt, pred):
    anomaly_state = False
    for i in range(len(gt)):
        if gt[i] == 1 and pred[i] == 1 and not anomaly_state:
            anomaly_state = True
            for j in range(i, 0, -1):
                if gt[j] == 0:
                    break
                else:
                    if pred[j] == 0:
                        pred[j] = 1
            for j in range(i, len(gt)):
                if gt[j] == 0:
                    break
                else:
                    if pred[j] == 0:
                        pred[j] = 1
        elif gt[i] == 0:
            anomaly_state = False
        if anomaly_state:
            pred[i] = 1
    return gt, pred


def cal_accuracy(y_pred, y_true):
    return np.mean(y_pred == y_true)

from functools import lru_cache
from operator import itemgetter

import numpy as np
import pandas as pd
from sklearn import svm
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.utils import resample
from sympy import prime


class TraitsObjective(object):
    def __init__(self, prep_train, trait, eyes, frqs, cv, direction, to_test, score_fun):
        # self.data = eeg_data
        self.pt = prep_train
        self.trait = trait
        self.eyes = eyes
        self.frqs = frqs
        self.cv = cv
        assert direction == "max" or direction == "min" or direction == "maxmin"
        self.direction = direction
        assert to_test == 'train' or to_test == 'test' or to_test == 'error'
        self.to_test = to_test
        self.score_fun = score_fun
        # some default SVR param
        self.gamma = 'scale'
        self.epsilon = 0.01
        self.shrinking = True
        self.degree = 3.0

    def train_test(self, x, target, kernel='linear', degree=3.0, C=1.0, repeat=10, train_size=0.7):
        r2_train_t = np.zeros(repeat)
        r2_test_t = np.zeros(repeat)
        # r2_score_t = np.zeros(repeat)
        for k in range(repeat):
            x_train, x_test, y_train, y_test = train_test_split(x, target, train_size=train_size,
                                                                random_state=prime(3 * (k + 1)))
            svr = svm.SVR(kernel=kernel, degree=degree, gamma=self.gamma, C=C, epsilon=self.epsilon,
                          shrinking=self.shrinking, max_iter=-1)

            # info with no stanard scaling
            regr = svr
            # info with scaling
            # regr = make_pipeline(StandardScaler(), svr)
            regr.fit(x_train, y_train)
            r2_train_t[k] = regr.score(x_train, y_train)
            # y_hat_test = svr.predict(x_test)
            # r2_score_t[k] = r2_score(y_test, y_hat_test)
            r2_test_t[k] = regr.score(x_test, y_test)

        return r2_test_t  # , r2_train_t  # , r2_score_t

    def train_bootstrap(self, x, target, kernel='linear', degree=3.0, gamma='scale', C=1.0, epsilon=0.01,
                        shrinking=True, repeat=10):
        train_r2 = np.zeros(repeat)
        # test_r2 = np.zeros(repeat)
        test_err = np.zeros(repeat)
        test_corr = np.zeros(repeat)
        svr = svm.SVR(kernel=kernel, degree=degree, gamma=gamma, C=C, epsilon=epsilon,
                      shrinking=shrinking, max_iter=-1)
        for k in range(repeat):
            x_train, y_train, ind = resample(x, target, range(x.shape[0]), replace=True,
                                             random_state=prime(3 * (k + 1)))
            ind_train = sorted(list(set(ind)))
            ind_test = sorted(list(set(range(x.shape[0])) - set(ind_train)))

            # info with no stanard scaling
            # regr = svr
            # info with scaling
            regr = make_pipeline(StandardScaler(), svr)
            regr.fit(x_train, y_train)

            # info adjusted r_2 score (on train data)
            # train_r2[k] = 1 - (1 - regr.score(x[ind_train], target[ind_train])) * (len(ind_train) - 1) / \
            #               (len(ind_train) - x.shape[1] - 1)
            # test_r2[k] = 1 - (1 - regr.score(x[ind_test], target[ind_test])) * (len(ind_test) - 1) / \
            #              (len(ind_test) - x.shape[1] - 1)
            test_err[k] = np.sqrt(mean_squared_error(target[ind_test], regr.predict(x[ind_test])))
            test_corr[k] = self.score_fun(regr.predict(x[ind_test]), target[ind_test]).correlation

        if self.to_test == 'train':
            return train_r2, test_err
        elif self.to_test == 'test' or self.to_test == 'error':
            return test_corr, test_err
        else:
            raise ValueError(f'incorrect self.to_test = {self.to_test} value')

    @lru_cache(maxsize=31)
    def get_x_y(self, frq):
        self.pt.select_freq_bin(frq=frq)
        x_var, target = self.pt.get_X_y(predict_label=self.trait)
        return x_var, target

    def __call__train_test(self, trial):
        frq = trial.suggest_int(name="freq", low=1, high=self.frqs)
        kernel = trial.suggest_categorical("kernel", ["linear", 'rbf', 'poly'])
        if kernel == 'poly':
            degree = trial.suggest_discrete_uniform(name='degree', low=3, high=7, q=0.1)
        else:
            degree = 1.0
        C = trial.suggest_discrete_uniform(name="C", low=1, high=50, q=0.1)
        # gamma = trial.suggest_categorical("gamma", ["scale", "auto"])
        # epsilon = 0.05
        # shrinking = True

        x_var, target = self.get_x_y(frq=frq)
        score = self.train_test(x=x_var, target=target, kernel=kernel, degree=degree, C=C)
        score = score.mean()
        return score

    def __call__(self, trial):
        frq = trial.suggest_int(name="freq", low=1, high=self.frqs)
        kernel = trial.suggest_categorical("kernel", ["linear", 'rbf'])  # , 'poly'])
        # kernel = 'rbf'
        if kernel == 'poly':
            degree = trial.suggest_discrete_uniform(name='degree', low=3, high=7, q=0.1)
        else:
            degree = 1.0
        C = trial.suggest_discrete_uniform(name="C", low=1, high=50, q=0.1)
        # gamma = trial.suggest_categorical("gamma", ["scale", "auto"])
        gamma = "scale"
        epsilon = 0.05
        shrinking = True

        x_var, target = self.get_x_y(frq=frq)
        # todo zamiast accu liczyc r2 na test i porownywac
        score, err = self.train_bootstrap(x=x_var, target=target, kernel=kernel, degree=degree, gamma=gamma, C=C,
                                          epsilon=epsilon, shrinking=shrinking)
        score = score.mean()
        err = err.mean()
        # roc = np.sqrt((1 - score) ** 2 + err ** 2)
        # run['score'].log(score)
        # run['err'].log(min(2, err))
        # run['roc'].log(min(4, roc))
        if self.direction == "max":
            return score  # , err
        elif self.direction == "min":
            return err
        else:
            return score, err

    # test all best models on the given input
    def test_predictions(self, results, best_trials, test_data):
        prm = {'trait': [], 'eyes': [], 'freq': [], 'kernel': [], 'C': [], 'score': [], 'error': [], 'roc': []}
        for trial in best_trials:
            prm['trait'].append(self.trait)
            prm['eyes'].append(self.eyes)
            prm['freq'].append(trial.params['freq'])
            prm['kernel'].append(trial.params['kernel'])
            prm['C'].append(np.round(trial.params['C'], 2))
            prm['score'].append(np.round(trial.values[0], 5))
            prm['error'].append(np.round(trial.values[1], 5))
            roc = np.sqrt((1 - trial.values[0]) ** 2 + trial.values[1] ** 2)
            prm['roc'].append(np.round(roc, 5))
        params = pd.DataFrame.from_dict(prm)
        # info remove duplicate rows
        params = params.drop_duplicates(subset=['trait', 'eyes', 'freq', 'kernel', 'C'])
        dd = {'eyes': [], 'roc': [], 'freq': [], 'kernel': [], 'C': [], 'score': [], 'error': [], 'trait': [],
              'predict': []}
        new_results = pd.DataFrame(params)
        new_results['predict'] = 0
        for idx, trial in params.iterrows():
            freq, kernel, C = itemgetter('freq', 'kernel', 'C')(trial)
            x_var, target = self.get_x_y(frq=freq)  # info reads the train file, thus is OK
            svr = svm.SVR(kernel=kernel, degree=self.degree, gamma=self.gamma, C=C, epsilon=self.epsilon,
                          shrinking=self.shrinking, max_iter=-1)
            regr = make_pipeline(StandardScaler(), svr)
            regr.fit(x_var, target)
            test_frq = test_data[freq].to_numpy().reshape(-1, 19)
            predicted = regr.predict(test_frq)
            # new_results.loc[idx, 'predict'] = predicted

        # info new_results to the ones in params
        final_results = pd.concat([results, new_results])
        return final_results

    # info opcja z cross_val_score
    def __call__cross(self, trial):
        frq = trial.suggest_int(name="freq", low=1, high=self.frqs)
        kernel = trial.suggest_categorical("kernel", ["linear", 'rbf', 'poly'])
        if kernel == 'poly':
            degree = trial.suggest_discrete_uniform(name='degree', low=3, high=7, q=0.1)
        else:
            degree = 1.0
        C = trial.suggest_discrete_uniform(name="C", low=1, high=50, q=0.1)
        gamma = 'scale'
        epsilon = 0.1
        shrinking = True

        x_var, target = self.get_x_y(frq=frq)
        svr = svm.SVR(kernel=kernel, degree=degree, gamma=gamma, C=C, epsilon=epsilon,
                      shrinking=shrinking, max_iter=-1)
        score = cross_val_score(svr, x_var, target, n_jobs=-1, cv=self.cv)
        accuracy = score.mean()
        # accuracy = np.max([accuracy, -1])
        return accuracy

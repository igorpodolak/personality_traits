import argparse
import platform
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn import svm, preprocessing
from sklearn.metrics import r2_score
from sklearn.model_selection import cross_val_score, ShuffleSplit, cross_validate, train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sympy import prime

from utils.PrepareTrain import PrepareTrain

import matplotlib.pyplot as plt
import seaborn as sns


def train_test(x, y, kernel='linear', repeat=10, train_size=0.7):
    test_size = 1. - train_size
    r2_train_t = np.zeros(repeat)
    r2_test_t = np.zeros(repeat)
    # r2_score_t = np.zeros(repeat)
    for k in range(repeat):
        x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=train_size,
                                                            random_state=prime(3 * (k + 1)))
        svr = svm.SVR(kernel=kernel, gamma='scale', C=1.0, epsilon=0.1, shrinking=True, max_iter=-1)

        # info with no stanard scaling
        regr = svr
        # info with scaling
        # regr = make_pipeline(StandardScaler(), svr)
        regr.fit(x_train, y_train)
        r2_train_t[k] = regr.score(x_train, y_train)
        # y_hat_test = svr.predict(x_test)
        # r2_score_t[k] = r2_score(y_test, y_hat_test)
        r2_test_t[k] = regr.score(x_test, y_test)

    return r2_train_t, r2_test_t  # , r2_score_t


def cross_score(x, y, cv=5):
    svr = svm.SVR(kernel='linear', gamma='scale', C=1.0, epsilon=0.1, shrinking=True, max_iter=-1)
    score = cross_val_score(svr, x, y, cv=cv)
    return score


def main(opts):
    channels_no = 64
    pt = PrepareTrain(datadir=opts.datadir, state=opts.state, eyes=opts.eyes, surveyfile=opts.surveyfile,
                      logdir=opts.logdir, channels=channels_no)
    pt.read_data()
    cv_no = 5

    df = pd.DataFrame()
    res = np.zeros((pt.max_frq - pt.min_frq + 1, 2))
    predict_label = 'A'
    for kernel in ('linear', 'rbf'):  # , 'poly', 'sigmoid'):
        print(f"\nStart for kernel = {kernel}")
        for freq in range(pt.min_frq, pt.max_frq + 1):
            pt.select_freq_bin(frq=freq)
            x_var, target = pt.get_X_y(predict_label=predict_label)
            r2_trn, r2_tst = train_test(x=x_var, y=target, kernel=kernel, repeat=10)
            res[freq - 1, :] = np.mean(r2_tst), np.var(r2_tst)
            res_dict = {'trait': [predict_label] * r2_tst.shape[0],
                        'channels': [channels_no] * r2_tst.shape[0],
                        'kernel': [kernel] * r2_tst.shape[0],
                        'freq': [freq] * r2_tst.shape[0],
                        'r2': r2_tst}
            df_x = pd.DataFrame(data=res_dict)
            df = pd.concat([df, df_x])
            print(f"{freq:2d} ---> {res[freq - 1, 0]:.4f} +- {res[freq - 1, 1]:.4f}")
    pass

    # info reset index from 0
    df.reset_index(drop=True, inplace=True)
    sns.lineplot(x='freq', y='r2', hue='kernel', data=df[df['trait'] == predict_label])
    plt.tight_layout()
    plt.show()

    return
    score = cross_score(x=x_var, y=target, cv=10)

    regr = svm.LinearSVR(max_iter=100000)
    scr1 = cross_val_score(regr, x_var, target, cv=cv_no, verbose=1)
    print(f"Scores {scr1.mean():.3f} accuracy with a standard deviation of {scr1.std():.3f}")

    regr = svm.LinearSVR(max_iter=100000)
    cv_split = ShuffleSplit(n_splits=cv_no, test_size=0.3, random_state=0)
    scr2 = cross_val_score(regr, x_var, target, cv=cv_split, verbose=1)
    print(f"Scores {scr2.mean():.3f} accuracy with a standard deviation of {scr2.std():.3f}")

    regr = svm.LinearSVR(max_iter=100000)
    clf = make_pipeline(preprocessing.StandardScaler(), regr)
    scr3 = cross_val_score(clf, x_var, target, cv=cv_no, verbose=1)
    print(f"Scores {scr3.mean():.3f} accuracy with a standard deviation of {scr3.std():.3f}")

    regr = svm.LinearSVR(max_iter=100000)
    scr4 = cross_validate(regr, x_var, target, cv=cv_no)
    print(f"cv results: {scr4['test_score']}")

    regr = svm.LinearSVR(max_iter=100000)
    scr5 = cross_validate(regr, x_var, target, cv=cv_no, scoring=('r2'))
    # print(f"cv results: {scr5['test_r2']}")

    svr = svm.SVR(kernel='linear', gamma='scale', C=1.0, epsilon=0.1, shrinking=True, max_iter=-1)
    clf = make_pipeline(preprocessing.StandardScaler(), svr)
    svr_r2 = cross_val_score(clf, x_var, target, cv=cv_no, verbose=1)
    print(f"Scores {svr_r2} accuracy with a standard deviation of {scr3.std():.3f}")

    # mse = 0.
    # regr = svm.LinearSVR(max_iter=100000)
    # n = 75
    # regr.fit(x_var[:n], target[:n])
    # for k in range(x_var.shape[0]):
    #     y = regr.predict(x_var[k].reshape(1, -1))[0]
    #     err = np.abs(y - target[k])
    #     mse += err
    #     print(f"{k:3d}: {y:.3f} vs {target[k]:.3f} wanted err={err:.5f}")
    # print(f"Total abs-err {mse:.5f}, mean {(mse / target.shape[0]):.5f}")

    pass


if __name__ == '__main__':
    if platform.node().startswith('Igors-MacBook-Pro') or platform.node().startswith('igor-podolak-6.laptop.matinf'):
        DATA_ROOT_DIR = '/Users/igor/data/'

    Datadir = f"{DATA_ROOT_DIR}/personality_traits/Data_Jach/Preprocessed_data"
    Surveydir = f"{DATA_ROOT_DIR}/personality_traits/Data_Jach/Survey"
    Surveyfile = "RestingEEG_BFAS.csv"

    # todo change State and eyes into lists
    State = ["Pre", "Post"]
    Eyes = ["Eyes Open", "Eyes Closed"]
    # person = "4274BB8B"
    # person = "1DC47E36"
    # person = "4C369AD1"
    Logdir = './logdir'

    parser = argparse.ArgumentParser()
    parser.add_argument('--datadir', default=Datadir, help='data directory path')
    parser.add_argument('--state', default=State, help='State of experiment')
    parser.add_argument('--eyes', default=Eyes, help='Eyses State')
    parser.add_argument('--logdir', type=str, default=Logdir, help='Path to directory where save results')
    parser.add_argument('--surveyfile', type=str, default=Path(Surveydir) / Surveyfile, help='Path of survey file')
    # parser.add_argument('--model_dir', type=str, default=model_dir, help='Path to directory where save results')
    # parser.add_argument('--suffix', default='trp', help="model save name suffix")
    # parser.add_argument('--prefix', default='trp', help="directory model save prefix")
    #
    # parser.add_argument('--numeric', default=True, help='cast all possible categoric data to numeric')

    args = parser.parse_args()

    main(opts=args)

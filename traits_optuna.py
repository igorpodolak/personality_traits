import argparse
import platform
from pathlib import Path

import numpy as np
import optuna
# import pandas as pd
import pandas as pd
from neptune.new.types import File
from scipy.stats import spearmanr
from sklearn import svm, preprocessing
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.model_selection import cross_val_score, ShuffleSplit, cross_validate, train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.utils import resample
from sympy import prime

from utils.PrepareTrain import PrepareTrain

# import matplotlib.pyplot as plt
# import seaborn as sns

import neptune.new as neptune
import neptune.new.integrations.optuna as optuna_utils

from utils.TraitsOptunaObjective import TraitsObjective

# accuracy = np.max([accuracy, -1])


run = None
neptune_callback = None


# run = neptune.init(project='GMUM/personality-traits')
# run = neptune.init(project='igor.t.podolak/personality-traits', mode='offline', flush_period=30)
# neptune_callback = optuna_utils.NeptuneCallback(run)

def read_model_param(file):
    df = pd.read_csv(file)
    return df


if __name__ == '__main__':
    if platform.node().startswith('Igors-MacBook-Pro') or platform.node().startswith('igor-podolak-6.laptop.matinf'):
        DATA_ROOT_DIR = '/Users/igor/data/'

    Datadir = f"{DATA_ROOT_DIR}/personality_traits/Data_Jach/Preprocessed_data"
    Surveydir = f"{DATA_ROOT_DIR}/personality_traits/Data_Jach/Survey"
    Surveyfile = "RestingEEG_BFAS.csv"
    # todo change State and eyes into lists
    State = ["Pre", "Post"]
    Eyes = ["Eyes Open", "Eyes Closed"]

    rogala_dir = "/Volumes/Samsung_T5/data/personality_traits/eeg_rogala_power"
    rogala_file = "rest_19_channel_power.csv"
    # person = "4274BB8B"
    # person = "1DC47E36"
    # person = "4C369AD1"

    param_file = "param_e_any.csv"
    Logdir = './logdir'
    cv_no = 5
    channels = 19
    predict_label = 'N'
    direction = "maxmin"
    to_test = 'test'
    n_trials = 500

    parser = argparse.ArgumentParser()
    parser.add_argument('--datadir', default=Datadir, help='data directory path')
    parser.add_argument('--testdir', default=rogala_dir, help='test data directory path')
    parser.add_argument('--state', default=State, help='State of experiment')
    parser.add_argument('--eyes', default=Eyes, help='Eyses State')
    # todo separate file for writing params to
    parser.add_argument('--param', type=str, default=param_file, help='param file to read from/ write to')
    parser.add_argument('--logdir', type=str, default=Logdir, help='Path to directory where save results')
    parser.add_argument('--surveyfile', type=str, default=Path(Surveydir) / Surveyfile, help='Path of survey file')
    parser.add_argument('--channels', type=int, default=channels, help='Number of input EEG channels')
    parser.add_argument('--cv', type=int, default=cv_no, help="CV folds")
    parser.add_argument('--predict_label', type=str, default=predict_label, help="the trait to predict")
    parser.add_argument('--direction', type=str, default=direction,
                        help="directions of optimization: max | min | maxmin")
    parser.add_argument('--to_test', type=str, default=to_test, help="the file to test score on (train or test)")
    #
    # parser.add_argument('--numeric', default=True, help='cast all possible categoric data to numeric')

    args = parser.parse_args()
    print(
        f"{'*' * 60}\npython traits_optuna --channels={args.channels} --predict_label={args.predict_label}\n{'_' * 60}")

    # info load the dataset in advance for reusing it each trial execution
    pt = PrepareTrain(datadir=args.datadir, state=args.state, eyes=args.eyes, surveyfile=args.surveyfile,
                      logdir=args.logdir, channels=args.channels)
    pt.read_data()
    # test_data = pt.read_single_rogala_power_data(fn=Path(rogala_dir) / rogala_file)
    all_test_data = pt.read_several_rogala_power_data(dirname=args.testdir)
    trait_dict = {'C': 'conscientiousness', 'A': 'agreeablness', 'N': 'neuroticism', 'O': 'openness',
                  'E': 'extrovertism'}
    run = neptune.init(project='igor.t.podolak/personality-traits', flush_period=30, mode='offline')
    neptune_callback = optuna_utils.NeptuneCallback(run)

    # run['sys/tags'].add([f"trait = {args.predict_label}", f"channels = {args.channels}", "train_test"])
    run['sys/tags'].add(
        [f"{trait_dict[args.predict_label]}", f"chann={args.channels}", "boot", f"score-{args.to_test}"])

    score_fun = spearmanr
    eyes = 'any' if type(args.eyes) in [list, tuple] else args.eyes
    objective = TraitsObjective(prep_train=pt, trait=args.predict_label, eyes=eyes, frqs=pt.max_frq, cv=args.cv,
                                direction=args.direction, to_test=args.to_test, score_fun=score_fun)
    study = None
    if args.direction == "max" or args.direction == "min":
        if args.to_test == 'train' or args.to_test == 'test':
            # info maximize score
            study = optuna.create_study(direction="maximize")
        elif args.to_test == 'error':
            # info minimize error
            study = optuna.create_study(direction="minimize")
        study.optimize(objective, n_trials=n_trials, callbacks=[neptune_callback], n_jobs=3)
        print(study.best_trial)
    else:
        param = read_model_param(Path(args.logdir) / args.param)
        # info multi: maximize score, minimize error
        study = optuna.create_study(directions=["maximize", "minimize"])
        study.optimize(objective, n_trials=n_trials, n_jobs=3)
        print("Number of finished trials: ", len(study.trials))
        pareto_front = optuna.visualization.plot_pareto_front(study, target_names=["score", "err"])
        pareto_front.update_layout(title=f"{trait_dict[args.predict_label]} Pareto front ({args.channels} channels)")
        run['visuals/pareto'] = File.as_html(pareto_front)
        # pareto_front.show()

        # run test on the outside data using best models found
        results = pd.DataFrame()
        # results = objective.test_predictions(results=results, best_trials=study.best_trials, test_data=test_data)
        results = objective.test_predictions(results=results, best_trials=study.best_trials, test_data=all_test_data)
        results.to_csv(f"{args.logdir}/{predict_label}_pred_{n_trials}.csv", index=False)
        pass

#!/usr/bin/env bash

python traits_optuna.py --channels=64 --predict_label=C

python traits_optuna.py --channels=64 --predict_label=A

python traits_optuna.py --channels=64 --predict_label=N

python traits_optuna.py --channels=64 --predict_label=O

python traits_optuna.py --channels=64 --predict_label=E

python traits_optuna.py --channels=19 --predict_label=C

python traits_optuna.py --channels=19 --predict_label=A

python traits_optuna.py --channels=19 --predict_label=N

python traits_optuna.py --channels=19 --predict_label=O

python traits_optuna.py --channels=19 --predict_label=E


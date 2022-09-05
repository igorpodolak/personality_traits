import pandas as pd
import platform
from pathlib import Path
import torch

# manual seed & device
torch.manual_seed(139462371)
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# from datasets.py
segment_length = 2000
sfreq = 500
epoch_len = segment_length / sfreq
overlap = 0  # 0.5

Df = pd.read_csv("personality_57_rec.csv")

if platform.node().startswith('LAPTOP-0TK'):
    path = Path().absolute() / 'data'
elif platform.node().startswith('Igors-MacBook-Pro') or platform.node().startswith('igor-podolak-6.laptop.matinf'):
    DATA_ROOT_DIR = Path('/Users/igor/data')
    # info nazwa kartoteki z plikami -- wartosc w parametrze wywolania --datadir
    #     datadir = f"{DATA_ROOT_DIR}/personality_traits/RESTS_gr87"
    # datadir = DATA_ROOT_DIR / 'personality_traits' / 'RESTS_gr87'
    standardized_dir = DATA_ROOT_DIR / 'personality_traits' / 'RESTS_gr87_standardized'
    path = DATA_ROOT_DIR


# from conv_seq.py
if platform.node().startswith('LAPTOP-0TK'):
    path_seq = Path().absolute() / 'data' / 'REST_standardized'
elif platform.node().startswith('Igors-MacBook-Pro') or platform.node().startswith('igor-podolak-6.laptop.matinf'):
    DATA_ROOT_DIR = Path('/Users/igor/data')
    # info nazwa kartoteki z plikami -- wartosc w parametrze wywolania --datadir
    #     datadir = f"{DATA_ROOT_DIR}/personality_traits/RESTS_gr87"
    # datadir = DATA_ROOT_DIR / 'personality_traits' / 'RESTS_gr87'
    standardized_dir = DATA_ROOT_DIR / 'personality_traits' / 'RESTS_gr87_standardized'
    path_seq = standardized_dir

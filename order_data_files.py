import os
from os import listdir
from os.path import isfile, join
# segregacja na foldery
datadir = "/Volumes/Samsung_T5/data/personality_traits/RESTS"
files = [f for f in listdir(datadir) if isfile(join(datadir, f))]
folders = [f[:8] for f in files]
folders = list(set(folders))
for folder in folders:
    os.mkdir(join(datadir, folder))
for file in files:
    os.replace(join(datadir, file), join(datadir, file[0:8], file))

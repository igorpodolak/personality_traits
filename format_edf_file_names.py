from pathlib import Path
import re

pattern = r'(\w{8}_rest)_[T]?(\d\d?)'
datadir = Path("/Volumes/Samsung_T5/data/personality_traits/RESTS")
for dir_path in datadir.iterdir():
    if dir_path.is_dir():
        for file in (datadir / dir_path).iterdir():
            name = file.parts[-1]
            match = re.search(pattern, name)
            new_path = dir_path.joinpath(f"{match.group(1)}_T{int(match.group(2)):02d}.edf")
            # print(f"{file} ---> {new_path}")
            file.rename(new_path)



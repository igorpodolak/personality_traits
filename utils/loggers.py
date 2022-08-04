import neptune.new as neptune
from neptune.new.types import File
import numpy as np
import copy


# PROJECT_NAME = "GMUM/FruitFly"

class Logger():
    def __init__(self, flags, user="igor.t.podolak", project="fruit-fly", flush_period=30):

        self.is_neptune = False
        self.is_text = False
        self.is_syslog = False
        if flags.get('neptune') is not None and flags['neptune']:
            self.run = neptune.init(project=f'{user}/{project}', flush_period=flush_period)
            self.is_neptune = True
            # self.run["parameters"] = self.params

    def log(self, key, value):
        if self.is_text:
            print(f"{key}: {value}")
        if self.is_neptune:
            self.run[key].log(value)

    def add(self, taglist):
        if self.is_neptune:
            self.run['sys/tags'].add(taglist)

    def upload(self, key, value):
        if self.is_neptune:
            self.run[key].upload(value)

    def stop(self):
        if self.is_neptune:
            self.run.stop()

    def id(self):
        if self.is_neptune:
            return self.run._label
        else:
            return "FRUIT-XXX"

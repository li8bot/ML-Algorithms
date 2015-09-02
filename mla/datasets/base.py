# Author: rushter <me@rushter.com>

import os
import pandas as pd
import numpy as np


def get_filename(name):
    return os.path.join(os.path.dirname(__file__), name)


def load_iris():
    df = pd.read_csv(get_filename('data/iris.csv'))
    species = pd.unique(df['species']).tolist()
    y = df['species'].apply(lambda x: species.index(x)).astype(int)
    df.drop('species', axis=1, inplace=type)
    return df.values, y.values


def load_robust():
    df = pd.read_csv(get_filename('data/robust.csv'))
    return df.values, None

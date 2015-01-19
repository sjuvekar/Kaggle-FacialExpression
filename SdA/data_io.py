import csv
import json
import numpy as np
import os
import pandas as pd
import cPickle
import gzip
from sklearn import cross_validation

def get_paths():
    paths = json.loads(open("Settings.json").read())
    for key in paths:
        paths[key] = os.path.expandvars(paths[key])
    return paths

def parse_dataframeX(df):
    parse_cell = lambda cell: np.fromstring(cell, dtype=np.float, sep=" ")
    df = df.applymap(parse_cell)
    ret = np.empty(shape=(len(df), 48*48))
    for i in range(0, len(df)):
	ret[i] = df.ix[i][0]
    return ret

def parse_dataframeY(df):
    parse_cell = lambda cell: int(cell)
    df = df.applymap(parse_cell)
    return np.array(df).flatten()

def read_train_data():
    train_path = get_paths()["train_path"]
    df = pd.read_csv(train_path)
    X = parse_dataframeX(df[["pixels"]])
    y = parse_dataframeY(df[["emotion"]])
    return (X, y)

def read_test_data():
    test_path = get_paths()["test_path"]
    df = pd.read_csv(test_path)
    return parse_dataframeX(df[["pixels"]])

def save_mnist(X, y):
    inX, testX, inY, testY = cross_validation.train_test_split(X, y, test_size=0.25)
    trainX, cvX, trainY, cvY = cross_validation.train_test_split(inX, inY, test_size=0.3333)
    data = [None, None, None]
    data[0] = (trainX, trainY)
    data[1] = (cvX, cvY)
    data[2] = (testX, testY)
    with gzip.open(get_paths()["mnist_path"], "w") as output:
	cPickle.dump(data, output) 

def save_model(model):
    out_path = get_paths()["model_path"]
    cPickle.dump(model, open(out_path, "w"))

def load_model():
    in_path = get_paths()["model_path"]
    return cPickle.load(open(in_path))

def read_submission():
    submission_path = get_paths()["submission_path"]
    return pd.read_csv(submission_path)

def write_submission(predictions):
    submission_path = get_paths()["submission_path"]
    writer = csv.writer(open(submission_path, "w"), lineterminator="\n")
    writer.writerows(rows)

import numpy
import pandas
import random
from matplotlib import pyplot as plt

if __name__ == "__main__":
    df = pandas.read_csv("../data/train.csv")

    emotions = {
        0: "Angry",
        1: "Disgust",
        2: "Fear",
        3: "Happy",
        4: "Sad",
        5: "Surprise"
    }

    for i in range(6):
        rand_idx = random.sample(df.index[df["emotion"] == 0], 1)
        print i, "->", rand_idx
        rand_img = df.ix[rand_idx]
        pixels = map(lambda a: float(a), rand_img["pixels"].values[0].split())
        plt.subplot(2, 3, i+1)
        plt.imshow(numpy.array(pixels).reshape(48, 48))
        plt.title(emotions[i])
    plt.show()

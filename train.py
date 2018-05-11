import random
import sys
import pickle
import glob
import copy

import numpy as np
from keras.callbacks import LambdaCallback
from keras.optimizers import SGD, RMSprop, Adam

from models import model, model2


encoder, autoencoder = model2()
buff = None

pkl_path = 'dataset/c_to_i.pkl'
data_path = 'dataset/corpus.distinct.txt'


def callbacks(epoch, logs):
    global buff
    buff = copy.copy(logs)
    print("epoch", epoch)
    print("logs", logs)


def train():
    with open(pkl_path, 'rb') as f:
        c_i = pickle.load(f)

    x, y = [], []
    with open(data_path, "r") as f:
        lines = [line for line in f]
        print(len(lines))
        random.shuffle(lines)
        for idx, line in enumerate(lines):
            print("now iter ", idx)
            if idx >= 150000:
                break
            line = line.strip()
            question, answer = line.split("___SP___")

            encoded_question = [[0.] * 128 for _ in range(50)]
            for i, c in enumerate(question):
                encoded_question[i][c_i[c]] = 1.
            # センテンスを逆順にして入れている(range(50)側)
            x.append(np.array(list(reversed(encoded_question))))

            encoded_answer = [[0.] * 128 for _ in range(50)]
            for i, c in enumerate(answer):
                encoded_answer[i][c_i[c]] = 1.
            y.append(np.array(encoded_answer))

    x = np.array(x)
    y = np.array(y)
    print(x.shape)
    if '--resume' in sys.argv:
        model = sorted(glob.glob("models/*.h5")).pop(0)
        print("loaded model is ", model)
        autoencoder.load_weights(model)

    for i in range(2000):
        print_callback = LambdaCallback(on_epoch_end=callbacks)
        batch_size = random.randint(32, 64)
        random_optim = random.choice([Adam(), SGD(), RMSprop()])
        # print(random_optim)
        autoencoder.optimizer = random_optim
        autoencoder.fit(x, y, shuffle=True, batch_size=batch_size, epochs=1, callbacks=[print_callback])
        autoencoder.save("models/%9f_%09d.h5" % (buff['loss'], i))
        print("saved ..")
        print("logs...", buff)


if __name__ == '__main__':
    train()

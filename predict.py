import pickle
import glob
import re

import numpy as np

from models import model
encoder, autoencoder = model()

c_to_i_path = 'dataset/c_to_i.pkl'
i_to_c_path = 'dataset/i_to_c.pkl'

data_path = 'dataset/corpus.distinct.txt'


def generate():
    c_i = pickle.loads(open(c_to_i_path, "rb").read())
    i_c = {i: c for c, i in c_i.items()}
    xss = []
    questions = []
    with open(data_path, "r") as f:
        for fi, line in enumerate(f):
            if fi >= 50:
                break
            line = line.strip()
            question, answer = line.split("___SP___")
            questions.append(question)
            xs = [[0.] * 128 for _ in range(50)]
            for i, c in enumerate(question):
                xs[i][c_i[c]] = 1.
            xss.append(np.array(list(reversed(xs))))
    Xs = np.array(xss)
    model = sorted(glob.glob("models/*.h5")).pop(0)
    print("loaded model is ", model)
    autoencoder.load_weights(model)

    Ys = autoencoder.predict(Xs).tolist()

    for ez, (question, y) in enumerate(zip(questions, Ys)):
        terms = []
        for v in y:
            term = max([(s, i_c[i]) for i, s in enumerate(v)], key=lambda x: x[0])[1]
            terms.append(term)
        answer = re.sub(r"」.*?$", "」", "".join(terms))
        print(ez, question, "___SP___", answer)
    # for sent_vector in Ys:
    #     encoded_sent = [np.argmax(vec) for vec in sent_vector]
    #     sent = [i_c[idx] for idx in encoded_sent]
    #     print(''.join(sent))


def predict():
    c_i = pickle.loads(open(c_to_i_path, "rb").read())
    i_c = {i: c for c, i in c_i.items()}
    xss = []
    questions = []
    with open(data_path, "r") as f:
        for fi, line in enumerate(f):
            if fi >= 50:
                break
            # print("now iter ", fi)
            line = line.strip()
            question, answer = line.split("___SP___")
            questions.append(question)
            xs = [[0.] * 128 for _ in range(50)]
            for i, c in enumerate(question):
                xs[i][c_i[c]] = 1.
            xss.append(np.array(list(reversed(xs))))

    Xs = np.array(xss)
    # print(Xs)
    model = sorted(glob.glob("models/*.h5")).pop(0)
    print("loaded model is ", model)
    autoencoder.load_weights(model)

    Ys = autoencoder.predict(Xs).tolist()
    for idx, (question, y) in enumerate(zip(questions, Ys)):
        terms = []
        for v in y:
            term = max([(s, i_c[i]) for i, s in enumerate(v)], key=lambda x: x[0])[1]
            terms.append(term)
        answer = re.sub(r"」.*?$", "」", "".join(terms))
        print(idx, question, "___SP___", answer)


if __name__ == '__main__':
    predict()

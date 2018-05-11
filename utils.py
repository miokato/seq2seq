import pickle
import os


def create_char_dict(path: str):
    """
    セパレータ「__SP__」を使って質問と回答を分割したテキストファイルを受け取り、
    テキストファイルに含まれる文字とインデックスの辞書をつくり、pickleで保存する
    param (str) path : 入力として用いるテキストファイルのパス
    """
    directory, input_path = os.path.split(path)
    ci_out_path = os.path.join(directory, 'c_i.pkl')
    ic_out_path = os.path.join(directory, 'i_c.pkl')

    with open(path) as f:
        sents = f.readlines()

    chars_of_question = []
    for sent in sents:
        question, answer = sent.split('__SP__')
        chars_of_question += [c for c in answer]
        chars_of_question += [c for c in question]

    chars_of_question = list(set(chars_of_question))
    i_to_c = {i: c for i, c in enumerate(chars_of_question)}
    c_to_i = {c: i for i, c in enumerate(chars_of_question)}

    with open(ci_out_path) as f:
        pickle.dump(c_to_i, f)

    with open(ic_out_path, 'wb') as f:
        pickle.dump(i_to_c, f)

import numpy as np
from copy import deepcopy
import json

c_e_dict = {}
c_e_f_dict = {}
c_j_i_l_m_dict = {}
c_i_l_m_dict = {}

t_f_e_dict = {}


def load_data(dataset="train", num_examples=False):
    en_data = []
    fr_data = []
    en_path = ""
    fr_path = ""

    ticker = 0

    if dataset == "train":
        en_path = "./training/hansards.36.2.e"
        fr_path = "./training/hansards.36.2.f"

    with open(en_path) as f:
        for line in f:
            l = format_line(line)
            l.insert(0, "null_word")

            en_data.append(l)

            if num_examples and num_examples < ticker:
                ticker = 0
                f.close()
                break

            ticker += 1

        f.close()

    with open(fr_path) as f:
        for line in f:
            fr_data.append(format_line(line))

            if num_examples and num_examples < ticker:
                f.close()
                break

            ticker += 1

        f.close()

    return [(en_data[i], fr_data[i]) for i in range(len(en_data))]


def format_line(line):
    return line.strip().split(" ")


def get_c_e_f(e_j, f_i):
    global c_e_f_dict

    num_times = 0

    if e_j in c_e_f_dict:
        if f_i in c_e_f_dict[e_j]:
            num_times = c_e_f_dict[e_j][f_i]

    return num_times


def get_c_e(e_j):
    global c_e_dict

    num_times = 0

    if e_j in c_e_dict:
        num_times = c_e_dict[e_j]

    return num_times


def reset_c_e_dict():
    global c_e_dict

    for e_j in c_e_dict:
        c_e_dict[e_j] = 0


def reset_c_e_f_dict():
    global c_e_f_dict

    for e_j in c_e_f_dict:
        for f_i in c_e_f_dict[e_j]:
            c_e_f_dict[e_j][f_i] = 0


def set_c_e_f(e_j, f_i, val):
    global c_e_f_dict

    if e_j in c_e_f_dict:
        if f_i in c_e_f_dict[e_j]:
            c_e_f_dict[e_j][f_i] = val


def set_c_e(e_j, val):
    global c_e_dict

    if e_j in c_e_dict:
        c_e_dict[e_j] = val


def get_t_f_e(e_j, f_i):
    global t_f_e_dict

    val = 0

    if e_j in t_f_e_dict and f_i in t_f_e_dict[e_j]:
        val = t_f_e_dict[e_j][f_i]

    return val


def set_t_f_e():
    global t_f_e_dict

    for e_j in t_f_e_dict:
        for f_i in t_f_e_dict[e_j]:
            t_f_e_dict[e_j][f_i] = get_c_e_f(e_j, f_i) / get_c_e(e_j)


def kroneckers_delta(sentence_pair, i, j):

    e = sentence_pair[0]
    f = sentence_pair[1]

    numerator_t_f_e = get_t_f_e(e[j], f[i])
    denominator_t_f_e = sum([get_t_f_e(e_j, f[i]) for e_j in e])

    return numerator_t_f_e / denominator_t_f_e


def create_global_dicts(data):
    base_dict = {}

    for pair in data:
        e = pair[0]
        f = pair[1]

        # base dict for use in c_e/c_e_f
        for e_j in e:
            if e_j not in base_dict:
                base_dict[e_j] = {}

            for f_i in f:
                if f_i not in base_dict[e_j]:
                    base_dict[e_j][f_i] = 0

    create_c_e_dicts(base_dict)
    create_t_f_e_dict(base_dict)


def create_c_e_dicts(base_dict):
    global c_e_dict
    global c_e_f_dict

    d1 = deepcopy(base_dict)
    d2 = deepcopy(base_dict)

    # c_e dict
    for e_j in d1:
        d1[e_j] = 0

    c_e_dict = d1

    # c_e_f dict
    c_e_f_dict = d2


def create_t_f_e_dict(base_dict):
    global t_f_e_dict

    base_dict = dict(base_dict)

    initital_value = 1/230

    for e_j in base_dict:
        for f_i in base_dict[e_j]:
            base_dict[e_j][f_i] = initital_value

    t_f_e_dict = base_dict


def save_t_f_e():
    global t_f_e_dict

    with open('t_f_e.json', 'w') as f:
        json.dump(t_f_e_dict, f)
        f.close()


def main():
    data = load_data(num_examples=1000)
    create_global_dicts(data)

    num_iterations = 10

    for iteration in range(num_iterations):
        print("iteration: " + str(iteration))
        # reset counters
        reset_c_e_dict()
        reset_c_e_f_dict()

        # iterate over sentence pairs
        for pair in data:
            e = pair[0]
            f = pair[1]

            l = len(e)
            m = len(f)

            # for word f_i in the french sentence
            for i in range(m):

                # for word e_j in the english sentence
                for j in range(l):
                    # kroneckers delta
                    k_delta = kroneckers_delta(pair, i, j)

                    # update c_e_f dict
                    new_c_e_f = get_c_e_f(e[j], f[i]) + k_delta
                    set_c_e_f(e[j], f[i], new_c_e_f)

                    # update c_e dict
                    new_c_e = get_c_e(e[j]) + k_delta
                    set_c_e(e[j], new_c_e)

        set_t_f_e()

    save_t_f_e()


main()

import numpy as np
from copy import deepcopy
import json

c_e_dict = {}
c_e_f_dict = {}

c_j_i_l_m_dict = {}
c_i_l_m_dict = {}
q_j_i_l_m_dict = {}

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


def reset_c_e_dict():
    global c_e_dict

    for e_j in c_e_dict:
        c_e_dict[e_j] = 0


def reset_c_e_f_dict():
    global c_e_f_dict

    for e_j in c_e_f_dict:
        for f_i in c_e_f_dict[e_j]:
            c_e_f_dict[e_j][f_i] = 0


def reset_c_j_i_l_m_dict():
    global c_j_i_l_m_dict

    for l in c_j_i_l_m_dict:
        for m in c_j_i_l_m_dict[l]:
            for j in c_j_i_l_m_dict[l][m]:
                for i in c_j_i_l_m_dict[l][m][j]:
                    c_j_i_l_m_dict[l][m][j][i] = 0


def reset_c_i_l_m_dict():
    global c_i_l_m_dict

    for l in c_i_l_m_dict:
        for m in c_i_l_m_dict[l]:
            c_i_l_m_dict[l][m] = 0


def get_c_e_f(e_j, f_i):
    global c_e_f_dict

    num_times = 0

    if e_j in c_e_f_dict:
        if f_i in c_e_f_dict[e_j]:
            num_times = c_e_f_dict[e_j][f_i]

    return num_times


def set_c_e_f(e_j, f_i, val):
    global c_e_f_dict

    if e_j in c_e_f_dict:
        if f_i in c_e_f_dict[e_j]:
            c_e_f_dict[e_j][f_i] = val


def get_c_e(e_j):
    global c_e_dict

    num_times = 0

    if e_j in c_e_dict:
        num_times = c_e_dict[e_j]

    return num_times


def set_c_e(e_j, val):
    global c_e_dict

    if e_j in c_e_dict:
        c_e_dict[e_j] = val


def get_c_i_l_m(l, m):
    global c_i_l_m_dict

    val = 0

    if l in c_i_l_m_dict and m in c_i_l_m_dict[l]:
        val = c_i_l_m_dict[l][m]

    return val


def set_c_i_l_m(l, m, val):
    global c_i_l_m_dict

    if l in c_i_l_m_dict and m in c_i_l_m_dict[l]:
        c_i_l_m_dict[l][m] = val


def get_c_j_i_l_m(l, m, j, i):
    global c_j_i_l_m_dict

    num_times = 0

    if l in c_j_i_l_m_dict:
        if m in c_j_i_l_m_dict[l]:
            if j in c_j_i_l_m_dict[l][m]:
                if i in c_j_i_l_m_dict[l][m][j]:
                    num_times = c_j_i_l_m_dict[l][m][j][i]

    return num_times


def set_c_j_i_l_m(l, m, j, i, val):
    global c_j_i_l_m_dict

    if l in c_j_i_l_m_dict:
        if m in c_j_i_l_m_dict[l]:
            if j in c_j_i_l_m_dict[l][m]:
                if i in c_j_i_l_m_dict[l][m][j]:
                    c_j_i_l_m_dict[l][m][j][i] = val


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


def get_q_j_i_l_m(l, m, j, i):
    global q_j_i_l_m_dict

    num_times = 0

    if l in q_j_i_l_m_dict and m in q_j_i_l_m_dict[l] and j in q_j_i_l_m_dict[l][m] and i in q_j_i_l_m_dict[l][m][j]:
        num_times = q_j_i_l_m_dict[l][m][j][i]

    return num_times


def set_q_j_i_l_m():
    global q_j_i_l_m_dict

    for l in q_j_i_l_m_dict:
        for m in q_j_i_l_m_dict[l]:
            for j in q_j_i_l_m_dict[l][m]:
                for i in q_j_i_l_m_dict[l][m][j]:
                    numerator = get_c_j_i_l_m(l, m, j, i)
                    denominator = get_c_i_l_m(l, m)

                    q_j_i_l_m_dict[l][m][j][i] = numerator / denominator


def kroneckers_delta(sentence_pair, j, i):

    e = sentence_pair[0]
    f = sentence_pair[1]

    l = len(e)
    m = len(f)

    numerator = get_q_j_i_l_m(l, m, j, i) * get_t_f_e(e[j], f[i])
    denominator = sum([get_q_j_i_l_m(l, m, k, i) *
                       get_t_f_e(e[k], f[i]) for k in range(l)])

    return numerator / denominator


def create_global_dicts(data):
    global c_i_l_m_dict
    global c_j_i_l_m_dict
    global q_j_i_l_m_dict

    q_j_i_l_m_dict_initial_val = 1/230
    base_dict = {}

    for pair in data:
        e = pair[0]
        f = pair[1]

        # c_i_l_m, c_j_i_l_m, and q_j_i_l_m dict
        len_e = len(e)
        len_f = len(f)

        if len_e not in c_i_l_m_dict:
            c_i_l_m_dict[len_e] = {}
            c_j_i_l_m_dict[len_e] = {}
            q_j_i_l_m_dict[len_e] = {}

        if len_f not in c_i_l_m_dict[len_e]:
            c_i_l_m_dict[len_e][len_f] = 0
            c_j_i_l_m_dict[len_e][len_f] = {}
            q_j_i_l_m_dict[len_e][len_f] = {}

        for l in range(len_e):
            if l not in q_j_i_l_m_dict[len_e][len_f]:
                c_j_i_l_m_dict[len_e][len_f][l] = {}
                q_j_i_l_m_dict[len_e][len_f][l] = {}

            for m in range(len_f):
                if m not in q_j_i_l_m_dict[len_e][len_f][l]:
                    c_j_i_l_m_dict[len_e][len_f][l][m] = 0
                    q_j_i_l_m_dict[len_e][len_f][l][m] = q_j_i_l_m_dict_initial_val

        # base dict for use in c_e/c_e_f
        for e_j in e:
            if e_j not in base_dict:
                base_dict[e_j] = {}

            for f_i in f:
                if f_i not in base_dict[e_j]:
                    base_dict[e_j][f_i] = 0

    create_c_e_dicts(base_dict)
    load_t_f_e_dict()


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


def load_t_f_e_dict():
    global t_f_e_dict

    path = "t_f_e.json"
    json_str = ""

    with open(path) as f:
        json_str = f.read()
        f.close()

    t_f_e_dict = json.loads(json_str)


def main():
    data = load_data(num_examples=1000)
    create_global_dicts(data)

    num_iterations = 10

    for iteration in range(num_iterations):
        print("iteration: " + str(iteration))
        # reset counters
        reset_c_e_dict()
        reset_c_e_f_dict()
        reset_c_j_i_l_m_dict()
        reset_c_i_l_m_dict()

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
                    k_delta = kroneckers_delta(pair, j, i)

                    # update c_e_f dict
                    new_c_e_f = get_c_e_f(e[j], f[i]) + k_delta
                    set_c_e_f(e[j], f[i], new_c_e_f)

                    # update c_e dict
                    new_c_e = get_c_e(e[j]) + k_delta
                    set_c_e(e[j], new_c_e)

                    # update c_j_i_l_m dict
                    new_c_j_i_l_m = get_c_j_i_l_m(l, m, j, i) + k_delta
                    set_c_j_i_l_m(l, m, j, i, new_c_j_i_l_m)

                    # update c_i_l_m dict
                    new_c_i_l_m = get_c_i_l_m(l, m) + k_delta
                    set_c_i_l_m(l, m, new_c_i_l_m)

        set_t_f_e()
        set_q_j_i_l_m()


main()

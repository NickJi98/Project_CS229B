# Spatial distribution analysis

import numpy as np


def parse_runtest(txtfile):
    N, T, t_new, t_old = [], [], [], []

    with open(txtfile, 'r') as file:
        for line in file:
            # Split the line into segments based on comma and space
            segments = line.strip().split(',')

            # Extract values for N, T, New method time, and Old method time
            N.append(int(segments[2].split()[1]))
            T.append(int(segments[3].split()[1]))
            t_new.append(float(segments[4].split()[2]))
            t_old.append(float(segments[5].split()[2]) if segments[5].split()[2] != 'failed' else np.nan)

    return np.array(N), np.array(T), np.array(t_new), np.array(t_old)


def parse_testfiles(txtfiles):
    N, T, t_new, t_old = [], [], [], []

    for txtfile in txtfiles:
        _N, _T, _t_new, _t_old = parse_runtest(txtfile)
        N.append(_N)
        T.append(_T)
        t_new.append(_t_new)
        t_old.append(_t_old)

    return np.concatenate(N), np.concatenate(T), np.concatenate(t_new), np.concatenate(t_old)

import numpy as np
import random
from itertools import product
def run_viterbi(emission_scores, trans_scores, start_scores, end_scores):
    """Run the Viterbi algorithm.

    N - number of tokens (length of sentence)
    L - number of labels

    As an input, you are given:
    - Emission scores, as an NxL array
    - Transition scores (Yp -> Yc), as an LxL array
    - Start transition scores (S -> Y), as an Lx1 array
    - End transition scores (Y -> E), as an Lx1 array

    You have to return a tuple (s,y), where:
    - s is the score of the best sequence
    - y is the size N array/seq of integers representing the best sequence.
    """
    L = start_scores.shape[0]
    assert end_scores.shape[0] == L
    assert trans_scores.shape[0] == L
    assert trans_scores.shape[1] == L
    assert emission_scores.shape[1] == L
    N = emission_scores.shape[0]

    y = []
    s = -np.inf
    sm = [[0 for i in xrange(L)] for j in xrange(N)]
    labl = [[0 for i in xrange(L)] for j in xrange(N)]
    # to include start_scores
    for i in xrange(L):
        sm[0][i] = start_scores[i] + emission_scores[0][i]
        labl[0][i]=i
    # viterbi calculation
    # T(i,y)=emmission_score(y,i,x)+max(transition_score(y',y)+T(y',i-1,x))
    for i in xrange(1, N):
        for j in xrange(L):
            sr = -np.inf
            maxl = 0
            for k in xrange(L):
                cal = sm[i - 1][k] + trans_scores[k][j]
                if cal > sr:
                    sr = cal
                    maxl = k

            sm[i][j] = sr + emission_scores[i][j]
            labl[i][j]=maxl
    # to include end scores
    for i in xrange(L):
        sm[N - 1][i] += end_scores[i]

    s = max(sm[N - 1][:])
    p = np.argmax(sm[N - 1][:])
    y.append(p)
    for i in xrange(N-1,0,-1):
        y.append(labl[i][p])
        p=labl[i][p]

    #print s, y
    return s,y[::-1]

'''''
    for y in product(range(L), repeat=N):  # all possible ys
        # compute its score
        score = 0.0
        score += start_scores[y[0]]
        for i in xrange(N - 1):
            score += trans_scores[y[i], y[i + 1]]
            score += emission_scores[i, y[i]]
        score += emission_scores[N - 1, y[N - 1]]
        score += end_scores[y[N - 1]]
        # update the best
        if score > s:
            s = score
            best_y = list(y)
    #print best_y
    '''''

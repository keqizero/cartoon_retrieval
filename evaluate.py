import numpy as np
import scipy.spatial as T

def test_recall1(cartoons, cartoon_names, portraits, portrait_names, dist_method='L2'):
    if dist_method == 'L2':
        dist = T.distance.cdist(cartoons, portraits, 'euclidean')
    elif dist_method == 'COS':
        dist = T.distance.cdist(cartoons, portraits, 'cosine')
    ord = dist.argsort()
    numcases = dist.shape[0]
    results = []
    for i in range(numcases):
        order = ord[i]
        results.append(portrait_names[order[0]])
    return results

def fx_calc_map_label(cartoons, cartoon_labels, portraits, portrait_labels, k = 0, dist_method='L2'):
    if dist_method == 'L2':
        dist = T.distance.cdist(cartoons, portraits, 'euclidean')
    elif dist_method == 'COS':
        dist = T.distance.cdist(cartoons, portraits, 'cosine')
    ord = dist.argsort()
    numcases = dist.shape[0]
    if k == 0:
        k = dist.shape[1]
    res = []
    for i in range(numcases):
        order = ord[i]
        p = 0.0
        r = 0.0
        for j in range(k):
            if all(cartoon_labels[i] == portrait_labels[order[j]]):
                r += 1
                p += (r / (j + 1))
        if r > 0:
            res += [p / r]
        else:
            res += [0]
    return np.mean(res)

def fx_calc_recall(cartoons, cartoon_labels, portraits, portrait_labels, recalls=[1, 5, 10], dist_method='L2'):
    if dist_method == 'L2':
        dist = T.distance.cdist(cartoons, portraits, 'euclidean')
    elif dist_method == 'COS':
        dist = T.distance.cdist(cartoons, portraits, 'cosine')
    ord = dist.argsort()
    numcases = dist.shape[0]
    k = dist.shape[1]
    results = []
    for recall in recalls:
        result = []
        for i in range(numcases):
            order = ord[i]
            r = 0
            r_a = 0
            for j in range(k):
                if all(cartoon_labels[i] == portrait_labels[order[j]]):
                    if j < recall:
                        r += 1
                    r_a += 1
                    if r_a >= recall:
                        break
            r = r / r_a
            result.append(r)
        results.append(np.mean(result))
    return results

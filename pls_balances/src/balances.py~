from sklearn.mixture import GaussianMixture
import tnumpy as np


def solve(w1, w2, m1, m2, std1, std2):
    # from stackoverflow
    # https://stackoverflow.com/a/22579904/1167475
    a = 1/(2*std1**2) - 1/(2*std2**2)
    b = m2/(std2**2) - m1/(std1**2)
    c = m1**2 /(2*std1**2) - m2**2 / (2*std2**2) - np.log((w1/w2) * np.sqrt(std2/std1))
    return np.roots([a,b,c])

def reorder(mid, m):
    lookup = {0: [1, 2], 1: [0, 2], 2: [0, 1]}
    l, r = lookup[mid]
    if m[l] > m[r]:
        l, r = r, l
    return l, mid, r

def round_balance(spectrum):
    gmod = GaussianMixture(n_components=3)
    gmod.fit(X=spectrum)
    m = gmod.means_
    std = np.sqrt(np.ravel(gmod.covariances_))
    w = gmod.weights_

    # first identify the distribution closest to zero
    mid = np.argmin(np.abs(m))

    # solve for intersections closest to zero
    l, mid, r = reorder(mid, m)
    lsol = solve(w[mid], w[l], m[mid], m[l], std[mid], std[l])
    rsol = solve(w[mid], w[r], m[mid], m[r], std[mid], std[r])

    lsol = lsol[np.argmin(np.abs(lsol))]
    rsol = rsol[np.argmin(np.abs(rsol))]
    #lsol, rsol = m[l][0], m[r][0]
    return lsol, rsol

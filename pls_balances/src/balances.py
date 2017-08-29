from skbio.stats.composition import clr, centralize
from sklearn.cross_decomposition import PLSRegression
from sklearn.mixture import GaussianMixture
from sklearn.model_selection import StratifiedKFold, KFold
from sklearn import preprocessing
from sklearn.metrics import roc_curve
from sklearn.metrics import auc
import numpy as np
import pandas as pd


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

def round_balance(spectrum, **init_kwds):
    """ Rounds a balance given single PLS component. """
    gmod = GaussianMixture(n_components=3, **init_kwds)
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
    if lsol<rsol:
        return lsol, rsol
    else:
        return rsol, lsol

def balance_classify(table, cats, num_folds, **init_kwds):
    """
    Builds a balance classifier. If categorical, it is assumed
    that the classes are binary.
    """
    skf = KFold(n_splits=num_folds, shuffle=True)

    ctable = pd.DataFrame(clr(centralize(table+1)),
                          index=table.index, columns=table.columns)

    cv = pd.DataFrame(columns=['Q2', 'AUROC'], index=np.arange(num_folds))
    for i, (train, test) in enumerate(skf.split(ctable.values, cats.values)):

        X_train, X_test = ctable.iloc[train], ctable.iloc[test]
        Y_train, Y_test = cats.iloc[train], cats.iloc[test]
        plsc = PLSRegression(n_components=1)
        plsc.fit(X=X_train, Y=Y_train)
        pls_df = pd.DataFrame(plsc.x_weights_, index=ctable.columns,
                              columns=['PLS1'])

        l, r = round_balance(pls_df, **init_kwds)
        denom = pls_df.loc[pls_df.PLS1 < l]
        num = pls_df.loc[pls_df.PLS1 > r]

        # make the prediction and evaluate the accuracy
        idx = table.index[test]
        pls_balance = (np.log(table.loc[idx, num.index] + 1).mean(axis=1) -
                       np.log(table.loc[idx, denom.index] + 1).mean(axis=1))

        group_fpr, group_tpr, thresholds = roc_curve(y_true=1-(Y_test==1).astype(int),
                                                     y_score=pls_balance)

        auroc = auc(group_tpr, group_fpr)
        press = ((pls_balance - Y_test)**2).sum()
        tss = ((Y_test.mean() - Y_test)**2).sum()
        Q2 = 1 - (press / tss)

        cv.loc[i, 'Q2'] = Q2
        cv.loc[i, 'AUROC'] = auroc

    # build model on entire dataset
    plsc = PLSRegression(n_components=1)
    plsc.fit(X=table.values, Y=cats.values)
    pls_df = pd.DataFrame(plsc.x_weights_, index=ctable.columns, columns=['PLS1'])
    l, r = round_balance(pls_df, **init_kwds)
    denom = pls_df.loc[pls_df.PLS1 < l]
    num = pls_df.loc[pls_df.PLS1 > r]
    pls_balance = (np.log(table.loc[:, num.index] + 1).mean(axis=1) -
                   np.log(table.loc[:, denom.index] + 1).mean(axis=1))

    return num, denom, pls_balance, cv

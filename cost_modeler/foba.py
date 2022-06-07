import loader
import logging
import matplotlib.ticker as mticker
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.optimize as sp
import statsmodels.api as sm


from typing import Optional, List, Pattern, Callable, Set, Iterable
from sklearn.cluster import KMeans
from sklearn import linear_model
from util import plot_cdf

pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)


class Regression:
    def __init__(self, A: np.matrix, Y: np.matrix):
        self.A = A.astype(float).reset_index(drop=True)
        self.A["b"] = 1
        self.Y = Y.astype(float).reset_index(drop=True)
        self.sol = None
        self.equation = None
        self.err = None

    def regress(self, weighted: bool = False):
        wA = self.A.copy()
        wY = self.Y.copy()
        #wA = (wA - wA.mean()) / (wA.std())
        #wA["b"] = 1
        if weighted:
            for index, row in wA.iterrows():
                if wY.loc[index][0] > 0:
                    wA.loc[index] *= 1 / (self.Y.iloc[index][0])
                    wY.loc[index][0] *= 1 / (self.Y.iloc[index][0])
        model = sm.OLS(wY.to_numpy().ravel(), wA.to_numpy())
        self.res = model.fit()
        self.sol = self.res.params
        #self.sol = model.fit_regularized().params
        self.sol = sp.nnls(wA.to_numpy(), wY.to_numpy().ravel())[0]
        #self.sol = sp.lsq_linear(wA.to_numpy(), wY.to_numpy().ravel()).x
        self.equation = pd.DataFrame(data=[self.sol], columns=wA.columns)

        # Get the error
        #df = wA.copy()
        df = self.A.copy()
        df["predicted"] = np.dot(self.A.to_numpy(), self.sol)
        df["actual"] = self.Y.iloc[:,0]
        df["error_percent"] = ((df["actual"] - df["predicted"]) / df["actual"]) * 100
        df["abs_error_percent"] = abs(((df["actual"] - df["predicted"]) / df["actual"]) * 100)
        df["error^2"] = ((df["actual"] - df["predicted"]) * (df["actual"] - df["predicted"]))
        self.err = df

    @staticmethod
    def _test(sol: pd.DataFrame, testA: pd.DataFrame, testY: pd.DataFrame):
        df = testA.copy()
        df["b"] = 1
        df["predicted"] = np.dot(df.to_numpy(), sol)
        df["actual"] = testY.iloc[:,0]
        df["error_percent"] = ((df["actual"] - df["predicted"]) / df["actual"]) * 100
        df["abs_error_percent"] = abs(((df["actual"] - df["predicted"]) / df["actual"]) * 100)
        df["error^2"] = ((df["actual"] - df["predicted"]) * (df["actual"] - df["predicted"]))
        return df

    def test(self, testA, testY):
        return Regression._test(self.sol, testA, testY)

    def mean_percent_error(self):
        return self.err["abs_error_percent"].mean()

    def cross_validate(self) -> pd.DataFrame:
        err = pd.DataFrame()
        for i in range(len(self.A)):
            newA = self.A.copy()
            newA = newA.drop(newA.index[[i]])
            newY = self.Y.copy()
            newY = newY.drop(newY.index[[i]])
            r = Regression(newA, newY)
            r.regress()
            testA = self.A.loc[[i]]
            testY = self.Y.loc[[i]]
            err = err.append(r.test(testA, testY))
        return err


def expand(df: pd.DataFrame, input_cols: List[str], degree: int) -> Set[str]:
    terms = input_cols.copy()
    for i in range(degree-1):
        new_terms = []
        for t1_idx in range(len(terms)):
            t1 = terms[t1_idx]
            for t2_idx in range(t1_idx+1, len(terms)):
                t2 = terms[t2_idx]
                new_term = "{}*{}".format(t1, t2)
                df[new_term] = df[t1] * df[t2]
                new_terms.append(new_term)
        terms.extend(new_terms)
    return set(terms)


def get_regression(df: pd.DataFrame, a_cols: Iterable[str], y_col: str):
    a = df[sorted(a_cols)].copy()
    # a = a.assign(b=1)
    y = df[[y_col]].copy()
    r = Regression(a, y)
    return r


def bootstrap(df: pd.DataFrame, a_cols: Iterable[str], y_col: str, n_samples=100):
    coefficients = pd.DataFrame()
    for i in range(n_samples):
        samples = df.sample(len(df), replace=True)
        r = get_regression(samples, a_cols, y_col)
        r.regress(weighted=True)
        coefficients = coefficients.append(r.equation)
    print(coefficients.quantile(q=0.05))
    print(coefficients.describe())


def lasso(l: loader.FilesLoader, a_cols: List[str], y: str):
    logging.info("a_cols: %s", a_cols)
    for i in range(1, 11):
        alpha = 0.1 * i
        clf = linear_model.Lasso(alpha=alpha)
        clf.fit(l.df[a_cols].to_numpy(), l.df[y])
        logging.info("alpha: %s\n coeff: %s, b: %f", alpha, clf.coef_, clf.intercept_)
    r = Regression(l.df[a_cols], l.df[y])
    r.sol = np.append(clf.coef_, clf.intercept_)
    r.equation = pd.DataFrame(data=[clf.coef_], columns=a_cols)
    return r

def foba(l: loader.FilesLoader, y: str, terms: List[str], use_mse=False, max_degree=2, min_delta=0.1):
    #terms = expand(l.df, l.input_cols, max_degree)
    F = {0: set()}
    errs = {0: 100}
    weighted = True
    if use_mse:
        weighted = False
        errs = {0: (l.df[y] * l.df[y]).mean()}
    regression = {0: None}
    delta = {}
    k = 0
    while True:
        min_err = float("inf")
        min_r = None
        min_term = None
        for c in terms:
            a_cols = F[k] | {c}
            r = get_regression(l.df, a_cols, y)
            r.regress(weighted=weighted)
            err = r.err["abs_error_percent"].mean()
            if use_mse:
                err = r.err["error^2"].mean()
            logging.info("term: %s, err: %.3f, eqn: %s", c, np.sqrt(err), r.equation)
            #print(r.equation)
            #skip = False
            #lower, upper = r.res.conf_int(0.95)[-1]
            #if lower < 0:
            #    skip = True
            #for lower, upper in r.res.conf_int(0.95):
            #    if lower < 0:
            #        skip = True
            #if skip:
            #    continue
            if err < min_err:
                min_err = err
                min_r = r
                min_term = c
        F[k + 1] = F[k] | {min_term}
        regression[k+1] = min_r
        errs[k+1] = min_err
        delta[k+1] = (errs[k] - min_err)
        logging.info("F[k+1]: %s\ndelta: %s", F[k+1], delta)
        if delta[k+1] <= min_delta and k > 0:
            #print("breaking, term: {}, err: {}".format(min_term, min_err))
            #print(F)
            #print(errs)
            break
        #print("forward")
        #print(min_r.equation)
        #print(min_err)
        k += 1
        logging.info("k: %d", k)
        while True:
            #print("backward")
            min_err = float("inf")
            min_term = None
            min_r = None
            for c in F[k]:
                r = get_regression(l.df, F[k] - {c}, y)
                r.regress(weighted=weighted)
                #sol = regression[k].equation.copy()
                #sol[c] = 0
                #df = pd.DataFrame()
                #df["predicted"] = np.dot(regression[k].A.to_numpy(), sol.to_numpy().ravel())
                #df["actual"] = regression[k].Y.iloc[:, 0]
                #df["abs_error_percent"] = abs(((df["actual"] - df["predicted"]) / df["actual"]) * 100)
                #df["error^2"] = (df["actual"] - df["predicted"]) * (df["actual"] - df["predicted"])
                #err = df["abs_error_percent"].mean()
                err = r.err["abs_error_percent"].mean()
                if use_mse:
                    err = r.err["error^2"].mean()
                #print("term: {}, err: {}".format(c, err))
                if err < min_err:
                    min_err = err
                    min_term = c
                    min_r = r
            logging.info("min_err: %.3f, min_term: %s, errs[k]: %.3f", min_err, min_term, errs[k])
            if k <= 1 or ((min_err - errs[k])) > (0.9* delta[k]):
                break
            logging.info("dropping term: %s", min_term)
            k -= 1
            F[k] = F[k+1] - {min_term}
            #r = get_regression(l.df, F[k], y)
            #r.regress(weighted=True)
            #regression[k] = r
            #err = r.err["abs_error_percent"].mean()
            #err = r.err["error^2"].mean()
            errs[k] = min_err
            delta[k] = errs[k-1] - min_err
            regression[k] = min_r
            #print("backward")
            #print(min_r.equation)
            #print(min_err)
    #print(regression[k-1].err)
    logging.info("deltas: %s\nk: %.3f\nF: %s", delta, k, F)
    return regression[k], sorted(list(F[k]))


def pbd(l: loader.FilesLoader, y: str):
    r = get_regression(l.df, l.input_cols, y)
    r.regress(weighted=False)
    logging.info(r.equation)
    logging.info(r.err)
    logging.info(r.res.summary())
    logging.info(r.res.conf_int(0.1))
    rt = r.err["abs_error_percent"]
    logging.info("p5: %.3f p50: %.3f p95: %.3f", rt.quantile(0.05), rt.quantile(0.5), rt.quantile(0.95))
    logging.info("RMSE: %.3f", np.sqrt((r.err["error^2"].mean())))
    return r


def ccd(l: loader.FilesLoader, y: str, terms: List[str], max_degree: int, min_delta: int = 1):
    #r, vars = foba(l, y, use_mse=True, min_delta=10, max_degree=max_degree)
    r, vars = foba(l, y, terms, use_mse=True, min_delta=min_delta, max_degree=max_degree)
    #print(r.equation)
    #print(r.err)
    #print(r.res.summary())
    return r, vars


def arith():
    l = loader.ABYLoader()
    l.load_files(**loader.ARITH)
    degree = 2
    degree = 1
    l.input_cols = ["i", "o", "ADD", "MUL", "w"]
    terms = l.input_cols
    lt = loader.ABYLoader()
    lt.load_files(**loader.ARITH_RAND)
    r, vars = ccd(l, "total_runtime", terms, max_degree=1)
    err = r.test(lt.df[vars], lt.df[["total_runtime"]])
    print(err)
    rt = err["abs_error_percent"]
    print(rt.quantile(0.05), rt.quantile(0.5), rt.quantile(0.95))
    print(np.sqrt((err["error^2"].mean())))

def main():
    arith()

def _main():
    l = loader.ABYLoader()
    l.load_files(**loader.ARITH)
    degree = 2
    degree = 1
    l.input_cols = ["i", "o", "XOR", "AND"]
    l = loader.ABYLoader()
    l.load_files(**loader.BOOL)
    l.input_cols = ["i", "o", "w", "XOR", "AND"]
    l = loader.ABYLoader()
    l.load_files(**loader.BOOL)
    metric = "func_indep_runtime"
    metric = "func_indep_runtime"
    metric = "memory"
    metric = "total_runtime"
    metric = "total_comm"
    l = loader.AgMpcLoader()
    l.load_files(**loader.AGMPC)
    l.input_cols = ["i", "XOR", "AND", "1/log(AND)"]
    #terms = expand(l.df, l.input_cols, degree)
    #l.input_cols = ["i", "o", "w", "ADD", "MUL"]
    #l.input_cols = ["i", "o", "XOR", "AND", "1/log(C)"]
    #l = arithmem()
    #print(l.df)
    #l.input_cols = ["i", "o", "w", "ADD", "MUL"]
    #l = loader.ABYLoader()
    #l.load_files(**loader.YAO_MEM)
    #l.input_cols = ["i", "o", "XOR", "AND"]
    #print(l.df)
    #l.input_cols = ["i", "o", "w", "AND", "XOR"]
    #l.input_cols = ["i", "o", "A2B"]
    #l.input_cols = ["i", "o", "w", "ADD", "MUL"]
    terms = l.input_cols
    print(l.df)
    r = pbd(l, metric)
    lt = loader.AgMpcLoader()
    lt.load_files(**loader.AGMPC_RAND)
    #lt = agmpc_rand()
    #lt = bool_rand()
    #lt = arith_rand()
    #lt = arithmem_rand()
    #lt = yao_rand()
    #lt = loader.ABYLoader()
    def oops():
        lt.load_files(**loader.AGMPC_RAND)
        lt.input_cols = ["i", "XOR", "AND", "1/log(AND)"]
        expand(lt.df, lt.input_cols, degree)
        lt.input_cols = l.input_cols
        lt.df['b'] = 1
        err = r.test(lt.df[r.equation.columns], lt.df[[metric]])
        print(err)
        pbd(lt, metric)
    #lt = loader.ABYLoader()
    #lt.load_files(**loader.BOOL_RAND)
    #lt.input_cols = ["i", "o", "AND", "XOR"]
    #l = arithmem()
    #l.input_cols = ["i", "o", "w", "ADD", "MUL"]
    #l = arith()
    #l.input_cols = ["i", "o", "w", "ADD", "MUL"]
    degree = 2
    #terms = expand(l.df, l.input_cols, degree)
    curr_min_delta = 2**25
    best_r = None
    best_vars = []
    curr_mean = float('inf')
    curr_vars = []
    best_vars = expand(l.df, terms, degree)
    best_r = lasso(l, best_vars, metric)
    """
    while True:
        _, vars = ccd(l, metric, terms, max_degree=degree, min_delta=curr_min_delta)
        if len(vars) == 1:
            break
        curr_min_delta *= 2
    r_min = 0
    r_max = curr_min_delta
    while r_max - r_min > 10:
        print(r_min, r_max, curr_min_delta)
        try:
            r, vars = ccd(l, metric, terms, max_degree=degree, min_delta=curr_min_delta)
        except Exception:
            r_max = curr_min_delta
            curr_min_delta = r_min + (r_max - r_min) / 2
            continue
        print(vars)
        if curr_vars == vars:
            r_max = curr_min_delta
            curr_min_delta = r_min + (r_max - r_min) / 2
            continue
        print(r.res.summary())
        print(r.res.pvalues)
        if any([pval > 0.05 for pval in r.res.pvalues[:-1]]):
            r_min = curr_min_delta
            curr_min_delta = r_min + (r_max - r_min) / 2
            continue
        c = r.cross_validate()
        #print(r.res.summary())
        m = np.sqrt(c["error^2"].mean())
        #m = c["abs_error_percent"].mean()
        print(curr_mean, m)
        #print(c.describe())
        print("imp.", (curr_mean - m) / curr_mean)
        if curr_mean == float('inf') or (curr_mean - m) / curr_mean > 0.05:
            print(r.err)
            best_r = r
            best_vars = vars
            curr_mean = m
            curr_vars = vars
            r_max = curr_min_delta
            curr_min_delta = r_min + (r_max - r_min) / 2
        else:
            r_min = curr_min_delta
            curr_min_delta = r_min + (r_max - r_min) / 2

    #pbd(lt, metric)
    best_r.regress(weighted=True)
    print(best_r.err)
    err = best_r.err
    """
    print(best_r.equation)
    err = best_r.test(l.df[best_vars], l.df[[metric]])
    print(err)
    rt = err["abs_error_percent"]
    print(rt.quantile(0.05), rt.quantile(0.5), rt.quantile(0.95))
    print(np.sqrt((err["error^2"].mean())))

    expand(lt.df, l.input_cols, degree=degree)
    print(lt.df.columns)
    err = best_r.test(lt.df[best_vars], lt.df[[metric]])
    err["XOR"] = lt.df["XOR"]
    err["o"] = lt.df["o"]
    print(err)
    rt = err["abs_error_percent"]
    print(rt.quantile(0.05), rt.quantile(0.5), rt.quantile(0.95))
    print(np.sqrt((err["error^2"].mean())))
    return
    err.to_csv("agmpc_{}".format(metric))

    l = bool()
    l.input_cols = ["i", "XOR", "AND", "o", "w"]
    lt = bool_rand()
    r, vars = ccd(l, "total_runtime", 1)
    err = r.test(lt.df[vars], lt.df[["total_runtime"]])
    print(err)
    rt = err["abs_error_percent"]
    print(rt.quantile(0.05), rt.quantile(0.5), rt.quantile(0.95))
    print(np.sqrt((err["error^2"].mean())))
    return

    l = yao()
    l.input_cols = ["i", "XOR", "AND", "o", "w"]
    lt = yao_rand()
    r, vars = ccd(l, "total_runtime", 1)
    err = r.test(lt.df[vars], lt.df[["total_runtime"]])
    print(err)
    rt = err["abs_error_percent"]
    print(rt.quantile(0.05), rt.quantile(0.5), rt.quantile(0.95))
    print(np.sqrt((err["error^2"].mean())))
    return
    l = agmpc()
    #l.input_cols = ["i", "i*1/log(C)", ]
    l.input_cols = ["i", "XOR", "AND", "1/log(C)"]
    expand(l.df, l.input_cols, 2)
    l.input_cols = ["o", "AND", "XOR", "AND*1/log(C)", "i*1/log(C)"]
    pbd(l, "total_comm")
    #r, vars = ccd(l, "total_comm", max_degree=2)
    return
    l = arith_apb()
    l.input_cols = ["i", "o", "MUL", "ADD", "w"]
    #pbd(l, "total_runtime")
    l = arith()
    print(l.df)
    l.input_cols = ["i", "o", "MUL", "ADD", "w"]
    #pbd(l, "total_runtime")
    r, vars = ccd(l, "total_runtime", max_degree=1)
    lt = arith_rand()
    lt.input_cols = ["i", "o", "MUL", "ADD", "w"]
    pbd(lt, "total_runtime")
    err = r.test(lt.df[vars], lt.df[["total_runtime"]])
    plot_cdf(err["abs_error_percent"])
    print(err)
    rt = err["abs_error_percent"]
    print(rt.quantile(0.05), rt.quantile(0.5), rt.quantile(0.95))
    print(np.sqrt((err["error^2"].mean())))
    return
    l = yao()
    l.input_cols = ["i", "o", "XOR", "AND", "w"]
    pbd(l, "online_comm")
    print(l.df.sort_values(l.input_cols))
    r, vars = ccd(l, "online_comm", max_degree=1)
    lt = bool_rand()
    lt.input_cols = ["i", "XOR", "AND"]
    pbd(lt, "online_comm")
    return
    err = r.test(lt.df[vars], lt.df[["total_comm"]])
    plot_cdf(err["abs_error_percent"])
    print(err)
    rt = err["abs_error_percent"]
    print(rt.quantile(0.05), rt.quantile(0.5), rt.quantile(0.95))
    return
    l = bool()
    l.input_cols = ["i", "o", "XOR", "AND", "w"]
    print(l.df.sort_values(l.input_cols))
    r, vars = ccd(l, "total_runtime", max_degree=1)
    lt = bool_rand()
    err = r.test(lt.df[vars], lt.df[["total_runtime"]])
    plot_cdf(err["abs_error_percent"])
    print(err)
    rt = err["abs_error_percent"]
    print(rt.quantile(0.05), rt.quantile(0.5), rt.quantile(0.95))
    return
    l = bool()
    l.input_cols = ["i", "o", "XOR", "AND", "w"]
    print(l.df.sort_values(l.input_cols))
    r, vars = ccd(l, "total_runtime", max_degree=1)
    lt = bool_rand()
    err = r.test(lt.df[vars], lt.df[["total_runtime"]])
    plot_cdf(err["abs_error_percent"])
    print(err)
    return
    l = bool_bpb()
    l.input_cols = ["i", "o", "XOR", "AND", "w"]
    pbd(l, "total_runtime")
    return
    l = agmpcmem_pbd()
    l.input_cols = ["i", "o", "XOR", "AND", "w"]
    pbd(l, "memory")
    l = agmpcmem()
    print(l.df)
    l.input_cols = ["i", "XOR", "AND", "1/log(C)"]
    ccd(l, "memory", max_degree=2)

    l = agmpc()
    l.input_cols = ["i", "XOR", "AND", "1/log(C)"]
    print(l.df)
    #expand(l.df, l.input_cols, 2)
    r, vars = foba(l, "func_dep_runtime", use_mse=False, max_degree=2, min_delta=3)
    print(r.equation)
    print(r.err)
    print(r.res.summary())
    return

    return
    l = yao_bpb()
    l.input_cols = ["i", "o", "XOR", "AND", "w"]
    pbd(l, "memory")
    l = yao()
    l.input_cols = ["i", "o", "XOR", "AND"]
    ccd(l, "memory", 1)
    return
    l = agmpc_pbd()
    l.input_cols = ["i", "o", "XOR", "AND", "w"]
    l.input_cols = ["i", "XOR", "AND"]
    r = get_regression(l.df, l.input_cols, "total_comm")
    r.regress(weighted=True)
    print(r.err)
    print(r.res.summary())
    print(r.res.conf_int(0.1))
    return

    l = yao_bpb()
    l.input_cols = ["i", "o", "XOR", "AND", "w"]
    r = get_regression(l.df, l.input_cols, "total_runtime")
    r.regress(weighted=True)
    print(r.err)
    print(r.res.summary())
    print(r.res.conf_int(0.1))
    l = arith_apb()
    l.input_cols = ["i", "o", "MUL", "ADD", "w"]
    #l = bool_bpb()
    #l.input_cols = ["i", "o", "XOR", "AND", "w"]
    r = get_regression(l.df, l.input_cols, "total_comm")
    r.regress(weighted=True)
    print(r.err)
    print(r.res.summary())
    print(r.res.conf_int(0.1))
    l = bool()
    l = yao()
    #l.input_cols = ["i", "MUL", "w"]
    l.input_cols = ["i", "o", "XOR", "AND", "w"]
    r, vars = foba(l, "total_runtime", use_mse=False, max_degree=1)
    print(r.equation)
    print(r.err)
    print(r.res.summary())
    #bootstrap(l.df, vars, "online_runtime")
    #print(r.res.summary())
    return
    test = arith_apb()
    expand(test.df, ["i", "w"], 1)
    test.df["b"] = 1
    print(r.test(test.df[r.equation.columns], test.df[["online_runtime"]]))

    return
    r = get_regression(l.df, ["i", "o", "MUL", "w"], "online_runtime")
    r.regress(weighted=False)
    print(r.equation)
    print(r.err)
    #l.df.plot(x="MUL", y="online_runtime")
    #plt.show()
    return
    foba(l, "online_runtime")
    return
    l = agmpc()
    foba(l, "online_runtime")
    return

    terms = expand(arith.df, arith.input_cols, 1)
    selected = []
    min_err = float("inf")
    min_r = None
    min_term = None
    delta = float("inf")
    while delta > 0:
        if min_term:
            terms.remove(min_term)
            selected.append(min_term)
            print(min_term)
            print(min_r.equation)
            print(min_err)
        prev_err = min_err
        for c in terms:
            A_cols = selected + [c]
            unique = arith.df.drop_duplicates(subset=A_cols+["online_comm"])
            #unique = arith.df
            A = unique[A_cols].copy()
            A = A.assign(b=1)
            Y = unique[["total_comm"]].copy()
            r = Regression(A, Y)
            r.regress(weighted=True)
            err = r.err["abs_error_percent"].mean()
            skip = False
            for lower, _ in r.res.conf_int(0.9):
                print(lower)
                if lower < 0:
                    skip = True
                    break
            if skip:
                continue
            print(c)
            print(err)
            if err < min_err:
                min_err = err
                min_r = r
                min_term = c
        delta = prev_err - min_err
    print(min_r.err)
    def split(r):
        err = r.err
        posA = err[err["error_percent"] >= 0][selected].copy()
        posA = posA.assign(b=1)
        posY = err[err["error_percent"] >= 0][["actual"]]
        pos_r = Regression(posA, posY)
        pos_r.regress(weighted=True)
        print(pos_r.equation)
        print(pos_r.mean_percent_error())
        print(pos_r.err)

        negA = err[err["error_percent"] < 0][selected].copy()
        negA = negA.assign(b=1)
        negY = err[err["error_percent"] < 0][["actual"]]
        neg_r = Regression(negA, negY)
        neg_r .regress(weighted=True)
        print(neg_r.equation)
        print(neg_r.mean_percent_error())
        print(neg_r.err)
        return pos_r, neg_r
    #plot_cdf(min_r.err["error_percent"])
    def split(r):
        kmeans = KMeans(n_clusters=2).fit(r.err["error_percent"].values.reshape(-1, 1))
        print(kmeans.labels_)
        labels = kmeans.labels_
        err = r.err
        posA = err[labels == 0][selected].copy()
        posA = posA.assign(b=1)
        posY = err[labels == 0][["actual"]]
        pos_r = Regression(posA, posY)
        pos_r.regress(weighted=True)
        print(pos_r.equation)
        print(pos_r.mean_percent_error())
        print(pos_r.err)

        negA = err[labels == 1][selected].copy()
        negA = negA.assign(b=1)
        negY = err[labels == 1][["actual"]]
        neg_r = Regression(negA, negY)
        neg_r .regress(weighted=True)
        print(neg_r.equation)
        print(neg_r.mean_percent_error())
        print(neg_r.err)
        return pos_r, neg_r
    pos_r, neg_r = split(min_r)
    split(pos_r)
    pos_r, neg_r = split(neg_r)
    return


    for c in ["n_input"]:
        A_cols = selected + [c]
        unique = arith.df.drop_duplicates(subset=A_cols+["total_comm"])
        A = unique[A_cols].copy()
        A = A.assign(b=1)
        Y = unique[["total_comm"]].copy()
        r = Regression(A, Y)
        r.regress(weighted=True)
        #plot_cdf(r.err["abs_error_percent"])
        #print(r.equation)
        print(r.err)
        print(c, "{:.4e}".format(r.err["error^2"].mean()))
        posA = r.err[r.err["error_percent"] >= 0][A_cols].copy()
        posA = posA.assign(b=1)
        posY = r.err[r.err["error_percent"] >= 0][["actual"]]
        negA = r.err[r.err["error_percent"] < 0][A_cols].copy()
        negA = negA.assign(b=1)
        negY = r.err[r.err["error_percent"] < 0][["actual"]]
        r = Regression(posA, posY)
        r.regress(weighted=True)
        print(r.err)
        print(c, "pos {:.4e}".format(r.err["error^2"].mean()))
        r = Regression(negA, negY)
        r.regress(weighted=True)
        print(r.err)
        print(c, "neg {:.4e}".format(r.err["error^2"].mean()))

        posA = r.err[r.err["error_percent"] >= 0][A_cols].copy()
        posA = posA.assign(b=1)
        print(posA)
        posY = r.err[r.err["error_percent"] >= 0][["actual"]]
        negA = r.err[r.err["error_percent"] < 0][A_cols].copy()
        negA = negA.assign(b=1)
        negY = r.err[r.err["error_percent"] < 0][["actual"]]
        r = Regression(posA, posY)
        r.regress(weighted=True)
        print(r.equation)
        print(r.err)
        print(c, "pos {:.4e}".format(r.err["error^2"].mean()))
        r = Regression(negA, negY)
        r.regress(weighted=True)
        print(r.equation)
        print(r.err)
        print(c, "neg {:.4e}".format(r.err["error^2"].mean()))


if __name__ == '__main__':
    main()

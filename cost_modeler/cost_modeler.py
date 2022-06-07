import sys
import logging
import os
import numpy as np

from pathlib import Path

from foba import pbd, ccd, expand, lasso
from loader import ABYLoader, AgMpcLoader
from logger import start_print_log, stop_print_log, BASIC_CONFIG

MAX_DEGREE = 2

def metric_to_dir(metric: str):
    if "runtime" in metric or "comm" in metric:
        return "runtime"
    elif "memory" in metric:
        return "memory"
    else:
        raise ValueError("Unsupported metric: %s" % metric)


def main():
    experiment_dir = Path(sys.argv[1])
    target_dir = experiment_dir.stem
    metric = sys.argv[2]
    out_dir = Path("models", target_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    log_file = str(Path(out_dir, "%s.log" % metric))
    logging.basicConfig(filename=log_file, **BASIC_CONFIG)

    logging.info("experiment dir: %s", experiment_dir)
    logging.info("target dir: %s", target_dir)
    logging.info("metric: %s", metric)

    #l = AgMpcLoader()
    l = ABYLoader()
    #experiment_dir = Path("logs", metric_to_dir(metric), "circuits", target_dir, "csv")
    experiment_dir = Path(experiment_dir, "csv")
    l.load_files([str(f) for f in experiment_dir.glob('*')], "ccd")
    logging.info(l.df)

    ##l.input_cols.remove("2")
    #l.input_cols.remove("w")
    #l.input_cols.remove("XOR")
    #l.input_cols.remove("o")
    #l.input_cols.append("1/log(C)")
    ##l.input_cols.append("C")

    r = pbd(l, metric)

    terms = l.input_cols
    print(l.input_cols)
    curr_min_delta = 2**25
    curr_vars = []
    curr_mean = float('inf')
    terms = expand(l.df, terms, MAX_DEGREE)
    best_r = lasso(l, terms, metric)
    while True:
        _, vars = ccd(l, metric, terms, max_degree=MAX_DEGREE, min_delta=curr_min_delta)
        if len(vars) == 1:
            break
        curr_min_delta *= 2
    r_min = 0
    r_max = curr_min_delta
    while r_max - r_min > 10:
        logging.info("r_min: %.3f r_max: %.3f, min_delta: %.3f", r_min, r_max, curr_min_delta)
        try:
            r, vars = ccd(l, metric, terms, max_degree=MAX_DEGREE, min_delta=curr_min_delta)
        except Exception:
            r_max = curr_min_delta
            curr_min_delta = r_min + (r_max - r_min) / 2
            continue
        logging.info("vars: %s", vars)
        if curr_vars == vars:
            r_max = curr_min_delta
            curr_min_delta = r_min + (r_max - r_min) / 2
            continue
        logging.info(r.res.summary())
        logging.info(r.res.pvalues)
        #if any([pval > 0.05 for pval in r.res.pvalues[:-1]]):
        #    r_min = curr_min_delta
        #    curr_min_delta = r_min + (r_max - r_min) / 2
        #    continue
        c = r.cross_validate()
        #print(r.res.summary())
        m = np.sqrt(c["error^2"].mean())
        #m = c["abs_error_percent"].mean()
        logging.info("curr_mean: %.3f iter_mean: %.3f", curr_mean, m)
        #print(c.describe())
        improvement = 1
        if curr_mean != float('inf'):
            improvement = (curr_mean - m) / curr_mean
        logging.info("improvement: %f", improvement)
        if improvement > 0.005:
            logging.info(r.err)
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
    start_print_log()
    logging.info(best_r.err)
    logging.info(best_r.equation)
    err = best_r.test(l.df[best_vars], l.df[[metric]])
    logging.info(err)
    rt = err["abs_error_percent"]
    logging.info("p5: %.3f p50: %.3f p95: %.3f", rt.quantile(0.05), rt.quantile(0.5), rt.quantile(0.95))
    logging.info("RMSE: %.3f", np.sqrt((err["error^2"].mean())))
    stop_print_log()

    model_file = Path(out_dir, "%s.model" % metric)
    best_vars.append("b")
    with open(model_file, "w") as f:
        for var in best_vars:
            print("%s: %f" % (var, best_r.equation[var]), file=f)



if __name__=='__main__':
    main()


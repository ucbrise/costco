import glob

import pandas as pd
import numpy as np

from loader.loader import FilesLoader

class AgMpcLoader(FilesLoader):
    def read_csv(self, f: str) -> pd.DataFrame:
        df = pd.read_csv(f)
        if "setup_runtime" in df.columns:
            df["setup_comm"] = df["setup_sent"] + df["setup_recv"]
            df["func_indep_comm"] = df["func_indep_sent"] + df["func_indep_recv"]
            df["func_indep_comm"] /= 1e3
            df["online_comm"] = df["online_sent"] + df["online_recv"]
            df["total_comm"] = df["setup_sent"] + df["setup_recv"] + df["func_indep_sent"] + \
                               df["func_indep_recv"] + df["func_dep_sent"] + df["func_dep_recv"] + \
                               df["online_sent"] + df["online_recv"]
            df["total_comm"] /= 1e3
            df["total_runtime"] = df["setup_runtime"] + df["func_indep_runtime"] + \
                              df["func_dep_runtime"] + df["online_runtime"]
        if "memory" in df.columns:
            df["memory"] /= 1e6
        return df

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        #df["log(C)"] = np.log(df["AND"] + df["i"])
        #df["log(C)"] = np.log(df["AND"] + df["i"] + df["XOR"] + df["o"])
        df["C"] = df["AND"] + df["XOR"]
        df["log(C)"] = np.log(df["AND"] + df["XOR"])
        #df["log(AND)"] = np.log(df["AND"])
        #df["1/log(AND)"] = 40 * df["log(AND)"].apply(lambda x: 1/x if x > 1 else 1)
        df["1/log(C)"] = df["log(C)"].apply(lambda x: 1/x if x > 1 else 1)
        #df["1/log(C)"] = 1 / max(np.log(df["AND"] + df["i"]), 1)
        #df["gates"] = df["AND"] + df["XOR"]
        return df

    def finalize(self):
        self.df = self.df.sort_values(["tag", "n_input", "n_and", "n_xor"]).reset_index(drop=True)


AGMPC_PBD = FilesLoader.load_files_args(
    files=glob.glob("agmpc-pbd-02-10/csv/*"),
    tag="pbd",
)

AGMPC = FilesLoader.load_files_args(
    glob.glob("agmpc-ccd-02-10/csv/*"),
    tag="ccd",
)

AGMPC_RAND = FilesLoader.load_files_args(
    glob.glob("agmpc-rand/csv/*"),
    tag="rand",
)

AGMPC_MEM_PBD = FilesLoader.load_files_args(
    glob.glob("bpbvalgrindresults/csv/*"),
    tag="ccd",
)


AGMPC_MEM_RAND = FilesLoader.load_files_args(
    glob.glob("boolrandvalgrindresults/csv/*"),
    tag="rand",
)

AGMPC_MEM = FilesLoader.load_files_args(
    glob.glob("ccdvalgrindresults/csv/*"),
    tag="ccd",
)

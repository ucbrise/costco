import glob

import pandas as pd

from loader.loader import FilesLoader


class ABYLoader(FilesLoader):
    @staticmethod
    def read_csv(f: str) -> pd.DataFrame:
        df = pd.read_csv(f)
        if "setup_runtime" in df.columns:
            df["total_runtime"] = df["setup_runtime"] + df["online_runtime"]
            df["total_comm"] = df["setup_comm"] + df["online_comm"]
        if "memory" in df.columns:
            df["memory"] /= 1e6
        return df

    @staticmethod
    def median_col(samples: pd.DataFrame, agg: pd.DataFrame):
        if "total_comm" in agg:
            agg["total_comm"] = samples["total_comm"].median()
            agg["setup_comm"] = samples["setup_comm"].median()

    def finalize(self):
        self.df = self.df.sort_values(["tag", "n_input", "n_mul", "n_add"])
        # self.df = self.df.drop("result", 1).reset_index(drop=True)


ARITH = FilesLoader.load_files_args(
    files=glob.glob("logs/circuits/arith_nodupes/csv/*"),
    tag="pbd-ccd",
)

ARITH_APB = FilesLoader.load_files_args(
    files=glob.glob("logs/circuits/apb/csv/*"),
    tag="pbd",
)

ARITH_RAND = FilesLoader.load_files_args(
    files=glob.glob("logs/circuits/arand/csv/*"),
    tag="rand",
)

ARITH_MEM_PBD = FilesLoader.load_files_args(
    files=glob.glob("logs/memory/circuits/apb/csv/*"),
    tag="pbd",
)

ARITH_MEM = FilesLoader.load_files_args(
    files=glob.glob("logs/memory/circuits/arith_nodupes/csv/*"),
    tag="pbd-ccd",
)


ARITH_MEM_RAND = FilesLoader.load_files_args(
    files=glob.glob("logs/memory/circuits/arand/csv/*"),
    tag="rand",
)

BOOL_BPB = FilesLoader.load_files_args(
    files=glob.glob("32bdoe/bpb/bool/*"),
    tag="pbd"
)

BOOL = FilesLoader.load_files_args(
    files=glob.glob("pbd_bccd_fixed/bool/*.csv"),
    tag="ccd",
)

BOOL_RAND = FilesLoader.load_files_args(
    files=glob.glob("32brand/bool/*"),
    tag="rand",
)

BOOL_MEM_PBD = FilesLoader.load_files_args(
    glob.glob("new-bool-yao-mem/bool/pbd/csv/*"),
    tag="pbd",
)

BOOL_MEM = FilesLoader.load_files_args(
    #glob.glob("logs/memory/circuits/yao_nodupes_fixed/csv/*"),
    glob.glob("new-bool-yao-mem/bool/bccd/csv/*"),
    tag="ccd",
)

BOOL_MEM_RAND = FilesLoader.load_files_args(
    glob.glob("new-bool-yao-mem/bool/rand/csv/*"),
    tag="rand",
)

A2Y_PBD = FilesLoader.load_files_args(
    glob.glob("logs/circuits/a2y_pbd/csv/*"),
    tag="pbd",
)

A2Y = FilesLoader.load_files_args(
    glob.glob("logs/circuits/a2y_ccd/csv/*"),
    tag="ccd",
)

B2Y_PBD = FilesLoader.load_files_args(
    glob.glob("logs/circuits/b2y_pbd/csv/*"),
    tag="pbd",
)

A2B = FilesLoader.load_files_args(
    glob.glob("logs/circuits/a2b_ccd/csv/*"),
    tag="ccd",
)

B2A = FilesLoader.load_files_args(
    glob.glob("logs/circuits/b2a_ccd/csv/*"),
    tag="ccd",
)

B2Y = FilesLoader.load_files_args(
    glob.glob("logs/circuits/b2y_ccd/csv/*"),
    tag="ccd",
)

Y2B_PBD = FilesLoader.load_files_args(
    glob.glob("logs/circuits/y2b_pbd/csv/*"),
    tag="pbd",
)

Y2B = FilesLoader.load_files_args(
    glob.glob("logs/circuits/y2b_ccd/csv/*"),
    tag="ccd",
)

Y2A = FilesLoader.load_files_args(
    glob.glob("logs/circuits/y2a_ccd/csv/*"),
    tag="ccd",
)

YAO_BPB = FilesLoader.load_files_args(
    glob.glob("32bdoe/bpb/yao/*"),
    tag="pbd",
)

YAO = FilesLoader.load_files_args(
    glob.glob("pbd_bccd_fixed/yao/*.csv"),
    tag="ccd",
)

YAO_RAND = FilesLoader.load_files_args(
    glob.glob("32brand/yao/*"),
    tag="rand",
)

YAO_MEM_PBD = FilesLoader.load_files_args(
    glob.glob("yao-mem-fixed/pbd/csv/*"),
    tag="pbd",
)

YAO_MEM = FilesLoader.load_files_args(
    #glob.glob("logs/memory/circuits/yao_nodupes_fixed/csv/*"),
    glob.glob("yao-mem-fixed/bccd/csv/*"),
    tag="ccd",
)

YAO_MEM_RAND = FilesLoader.load_files_args(
    glob.glob("yao-mem-fixed/rand/csv/*"),
    tag="rand",
)

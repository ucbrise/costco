import re

import pandas as pd

from typing import Dict, Iterable, List


FILE_RE = re.compile(r'(\d+)(\w+)[-\.]')


class FilesLoader:
    def __init__(self):
        self.df = pd.DataFrame()
        self.input_cols = []

    @staticmethod
    def load_files_args(files: Iterable[str], tag: str) -> Dict[str, object]:
        return {
            "files": files,
            "tag": tag,
        }

    @staticmethod
    def read_csv(f: str) -> pd.DataFrame:
        return pd.read_csv(f)

    @staticmethod
    def median_col(samples: pd.DataFrame, agg: pd.DataFrame):
        return

    @staticmethod
    def transform(df: pd.DataFrame) -> pd.DataFrame:
        return df

    def load_files(self, files: List[str], tag: str = None):
        for f in sorted(files):
            samples = self.read_csv(f)
            groups = re.findall(FILE_RE, f)
            if not self.input_cols:
                for _, input_col in groups:
                    self.input_cols.append(input_col)
            for num, input_col in groups:
                samples[input_col] = pd.to_numeric(num)
            samples = self.transform(samples)
            agg = samples.mean()
            self.median_col(samples, agg)
            if tag is not None:
                agg["tag"] = tag
            self.df = self.df.append([agg], ignore_index=True)



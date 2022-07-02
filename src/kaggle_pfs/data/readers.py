import glob

import pandas as pd

from kaggle_pfs import ROOT_DIR


def data_path(*args: str) -> str:
    path = (ROOT_DIR / ".." / ".." / "data").joinpath(*args)

    return str(path.resolve().absolute())


def sales_train_csv(filepath=None) -> pd.DataFrame:
    if filepath is None:
        filepath = data_path("raw", "sales_train.csv")

    df = pd.read_csv(filepath)

    df["date"] = pd.to_datetime(df["date"], format="%d.%m.%Y")

    return df


def items_by_month() -> pd.DataFrame:
    fl = glob.glob(data_path("processed", "items_by_month*", "part*.orc"))
    return pd.concat(map(pd.read_orc, fl), ignore_index=True)

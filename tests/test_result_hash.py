import pandas as pd

from nl2ocel.result_hash import hash_dataframe


def test_hash_dataframe_rounds_float_noise():
    left = pd.DataFrame({"stddev_days": [532.6877587297104]})
    right = pd.DataFrame({"stddev_days": [532.6877587297107]})

    assert hash_dataframe(left) == hash_dataframe(right)

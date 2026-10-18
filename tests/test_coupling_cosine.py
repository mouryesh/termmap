import numpy as np
import pandas as pd

from termmap.coupling import CodeEmbeddings


class _Enc:
    def encode(self, texts):
        return np.array([[1.0, 0.0] if t.startswith("alpha") else [0.0, 1.0] for t in texts], dtype=np.float32)


def test_cosine_embeds_codes_outside_the_primed_set():
    metas = {"ICD10CM": pd.DataFrame({"title": ["alpha one", "xyz two"]}, index=["A1", "B1"]),
             "SNOMEDCT_US": pd.DataFrame({"title": ["alpha"]}, index=["1"])}
    e = CodeEmbeddings(_Enc(), metas)
    e.prime([("ICD10CM", "A1"), ("SNOMEDCT_US", "1")])
    assert e.cosine("A1", "1") == 1.0
    assert e.cosine("B1", "1") == 0.0  # B1 was never primed: used to raise KeyError

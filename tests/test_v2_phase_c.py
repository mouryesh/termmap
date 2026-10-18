"""Phase C: ICD knowledge synonyms (C1)."""
import pandas as pd

from termmap.dictionaries import Dictionary
from termmap.icd_knowledge import index_terms
from termmap.rerank.cross_encoder import CandidateText
from termmap.retrieval.engine import description_table

XML = """<?xml version="1.0"?><ICD10CM.index><letter><title>F</title>
<mainTerm><title>Failure, failed</title>
  <term level="1"><title>heart<nemod>(acute)</nemod></title><code>I50.9</code>
    <term level="2"><title>congestive</title><code>I50.9</code></term>
    <term level="2"><title>stage B Note: long note</title><code>I50.9</code></term>
    <term level="2"><title>systolic</title><code>I50.2-</code></term></term></mainTerm>
<mainTerm><title>Cardiac failure</title><see>Failure, heart</see></mainTerm>
<mainTerm><title>Unknown thing</title><code>Z99.99</code></mainTerm>
</letter></ICD10CM.index>"""


def test_index_terms_paths_see_and_filters(tmp_path):
    p = tmp_path / "idx.xml"
    p.write_text(XML)
    got = set(index_terms(str(p), {"I50.9", "I50.20"})["description"])
    assert got == {"heart Failure", "congestive heart Failure", "stage B heart Failure", "Cardiac failure"}


def test_knowledge_rows_in_table_and_candidate_text():
    d = Dictionary("ICD10CM", pd.DataFrame({"code": ["I50.9"], "description": ["Heart failure, unspecified"]}))
    extra = pd.DataFrame({"code": ["I50.9", "I50.9", "I50.9"], "source": ["index", "index", "inclusion"],
                          "description": ["congestive heart Failure", "Heart failure, unspecified", "cardiac failure NOS"]})
    t = description_table(d, extra)
    assert t["source"].tolist() == ["inclusion", "index", "dict"]       # duplicate of the dict text dropped
    meta = pd.DataFrame({"title": ["Heart failure, unspecified"], "category_title": ["Heart failure"]},
                        index=pd.Index(["I50.9"], name="code"))
    ct = CandidateText("ICD10CM", t, meta)
    assert ct("I50.9") == "heart failure, unspecified | heart failure"
    assert ct("I50.9", "chf congestive heart failure") == \
        "heart failure, unspecified | congestive heart failure | heart failure"

"""Stage 0 LLM query rewrite (v2, item C2): one short standard concept name per (term, context).

Model: Qwen/Qwen3-8B (Apache-2.0), non-thinking mode, greedy decoding, at most 24 new tokens, fp16.
Outputs are cached by (term, sha1 of the context window) so training rows are generated once (Kaggle
job K1) and predict only runs the model for unseen pairs.
"""
from __future__ import annotations

import hashlib
import logging
import time

import pandas as pd

from termmap.rerank.cross_encoder import ENT, ENT_END, context_window

log = logging.getLogger(__name__)
MODEL = "Qwen/Qwen3-8B"
REVISION = "b968826d9c46dd6066d109eabc6255188de91218"  # Hugging Face commit, 2026-10-18
PROMPT = ("Rewrite this clinical term from a medical chart as the standard medical concept name; expand "
          "abbreviations; keep laterality, acuity, stage and negation; output only the name.")
MAX_NEW_TOKENS = 24


def llm_context(term_norm: str, chunk_text: str, chars: int = 100) -> str:
    """The cross-encoder context window with the term in quotes instead of [ENT] marks."""
    return context_window(term_norm, chunk_text, chars).replace(f"{ENT} ", '"').replace(f" {ENT_END}", '"')


def ctx_hash(ctx: str) -> str:
    return hashlib.sha1(ctx.encode()).hexdigest()[:16]


def user_message(term: str, ctx: str) -> str:
    return f"{PROMPT}\n\nTerm: {term}\nContext: {ctx}" if ctx else f"{PROMPT}\n\nTerm: {term}"


def clean_output(text: str) -> str:
    """First line, without quotes, a trailing period or a leftover empty think block."""
    text = text.replace("<think>", "").replace("</think>", "").strip()
    return text.split("\n")[0].strip().strip('"').rstrip(".").strip()


class Rewriter:
    def __init__(self, name: str = MODEL, device_map: str = "auto", revision: str | None = REVISION):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        self.tok = AutoTokenizer.from_pretrained(name, padding_side="left", revision=revision)
        self.model = AutoModelForCausalLM.from_pretrained(name, torch_dtype=torch.float16, device_map=device_map,
                                                          revision=revision)
        self.model.eval()

    def __call__(self, pairs: list[tuple[str, str]], batch_size: int = 32) -> list[str]:
        """Greedy rewrites, in input order; batches are fixed by input order for reproducibility."""
        import torch
        out: list[str] = []
        t0 = time.time()
        for s in range(0, len(pairs), batch_size):
            prompts = [self.tok.apply_chat_template([{"role": "user", "content": user_message(t, c)}], tokenize=False,
                                                    add_generation_prompt=True, enable_thinking=False)
                       for t, c in pairs[s:s + batch_size]]
            enc = self.tok(prompts, return_tensors="pt", padding=True).to(self.model.device)
            with torch.inference_mode():
                gen = self.model.generate(**enc, do_sample=False, max_new_tokens=MAX_NEW_TOKENS,
                                          pad_token_id=self.tok.pad_token_id)
            out.extend(clean_output(x) for x in self.tok.batch_decode(gen[:, enc["input_ids"].shape[1]:],
                                                                       skip_special_tokens=True))
            log.info("rewrite %d/%d (%.0fs)", len(out), len(pairs), time.time() - t0)
        return out


def unique_pairs(rows: pd.DataFrame, chars: int = 100) -> pd.DataFrame:
    """Unique (term, context) pairs of `rows`, sorted, with their cache key."""
    ctx = [llm_context(t, c, chars) for t, c in zip(rows["term_norm"], rows["chunk_text"])]
    df = pd.DataFrame({"term": rows["term_norm"].values, "ctx": ctx}).drop_duplicates()
    df["ctx_hash"] = df["ctx"].map(ctx_hash)
    return df.sort_values(["term", "ctx_hash"], kind="stable").reset_index(drop=True)

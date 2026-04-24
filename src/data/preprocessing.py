"""
============================================================================
 src/data/preprocessing.py  [REWRITE — v4]
 ---------------------------------------------------------------------------
 Publication-grade preprocessing pipeline for short-text topic modeling.

 v4 CHANGES (journal-grade)
 ───────────────────────────
   * AGGRESSIVE rare / frequent filtering to fix the poor U_Mass score:
       - `min_word_count` tightened per config
       - `max_word_freq`  tightened per config
       - filter_extremes applied as a hard boundary before BoW build.
   * Expanded domain-stopword list (discourse / social chatter).
   * Entropy filter tightened to 12 / 96 percentiles.
   * Saves contextual (sentence-transformer) document embeddings so the
     encoder can concat them with the BoW at train time.
   * preprocessing_report.json now records the contextual backbone + dim.
============================================================================
"""
from __future__ import annotations

import os
import re
import json
import pickle
from collections import Counter, defaultdict
from typing import List, Dict

import numpy as np
import nltk
from nltk.corpus import stopwords, wordnet
from nltk.stem   import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from nltk import pos_tag

from gensim.corpora import Dictionary
from gensim.models.phrases import Phrases, Phraser, ENGLISH_CONNECTOR_WORDS
from scipy import sparse

from ..utils.logging_utils import get_logger
from ..utils.contextual_embeddings import (
    load_contextual_doc_embeddings, DEFAULT_MODEL,
)

_LOG = get_logger(__name__)

# ─── NLTK resources ──────────────────────────────────────────────────────
for _pkg in ["punkt", "punkt_tab", "stopwords",
             "averaged_perceptron_tagger", "averaged_perceptron_tagger_eng",
             "wordnet", "omw-1.4"]:
    try:
        nltk.download(_pkg, quiet=True)
    except Exception:
        pass


# ─── regex + stopwords (expanded in v4) ──────────────────────────────────
_DOMAIN_STOPWORDS = {
    # web / social chatter
    "rt", "via", "amp", "http", "https", "www", "com", "org", "net",
    "tweet", "retweet", "twitter", "facebook", "instagram", "tiktok",
    "youtube", "reddit", "snapchat", "url", "link", "click",
    # reportage / discourse
    "said", "say", "says", "saying", "tell", "told", "telling",
    "would", "could", "should", "may", "might", "must", "shall",
    "also", "even", "still", "yet", "already", "however", "therefore",
    "actually", "really", "basically", "literally", "obviously",
    "maybe", "perhaps", "probably", "certainly", "definitely",
    # discourse scaffolding
    "one", "two", "three", "four", "five", "first", "second", "third",
    "last", "next", "previous", "former", "latter",
    "thing", "things", "way", "ways", "lot", "lots", "bit", "kind",
    "sort", "type", "case", "cases", "point", "piece",
    # generic high-freq verbs / fillers
    "get", "got", "getting", "go", "goes", "going", "gone", "went",
    "make", "made", "making", "take", "took", "taking", "takes",
    "come", "came", "coming", "comes",
    "like", "well", "back", "much", "many",
    "need", "needs", "want", "wants", "wanted",
    "know", "known", "knows", "knew",
    "think", "thought", "thinking", "thinks",
    "use", "used", "using", "uses",
    "see", "seen", "saw", "seeing",
    "look", "looks", "looked", "looking",
    # generic time
    "today", "tomorrow", "yesterday", "now", "then", "soon", "later",
    "hour", "hours", "minute", "minutes", "seconds",
    "week", "weeks", "month", "months", "year", "years",
    "day", "days", "morning", "evening", "night",
    # generic place / quantity
    "place", "area", "part", "parts", "side", "end", "beginning",
}
_SENTIMENT_KEEP = {"not", "no", "never", "good", "bad", "great", "poor",
                   "best", "worst", "high", "low", "new", "old"}

_URL_RE      = re.compile(r"http[s]?://\S+|www\.\S+")
_MENTION_RE  = re.compile(r"@\w+")
_HASHTAG_RE  = re.compile(r"#(\w+)")
_NUMBER_RE   = re.compile(r"\b\d+\b")
_MULTISPACE  = re.compile(r"\s+")
_NON_ALPHA   = re.compile(r"[^a-zA-Z\s]")


def _penn_to_wn(tag: str):
    if tag.startswith("J"): return wordnet.ADJ
    if tag.startswith("V"): return wordnet.VERB
    if tag.startswith("N"): return wordnet.NOUN
    if tag.startswith("R"): return wordnet.ADV
    return wordnet.NOUN


class PreprocessingPipeline:
    """End-to-end preprocessing driven by a Config.preprocessing block."""

    _PRIMARY_POS   = {"NN", "NNS", "NNP", "NNPS", "JJ", "JJR", "JJS"}
    _SECONDARY_POS = {"VB", "VBD", "VBG", "VBN", "VBP", "VBZ"}

    def __init__(self, cfg):
        self.cfg = cfg
        self.lemmatizer = WordNetLemmatizer()
        self.stopwords = (set(stopwords.words("english"))
                          | _DOMAIN_STOPWORDS) - _SENTIMENT_KEEP

    # ────────────────────────────────────────────────────────────────
    #  step 1 — normalisation & tokenisation
    # ────────────────────────────────────────────────────────────────
    def _normalize(self, text: str) -> str:
        text = text.lower()
        text = _URL_RE.sub(" ", text)
        text = _MENTION_RE.sub(" ", text)
        text = _HASHTAG_RE.sub(r"\1", text)
        text = _NUMBER_RE.sub(" ", text)
        text = _NON_ALPHA.sub(" ", text)
        return _MULTISPACE.sub(" ", text).strip()

    def _tokenize_and_lemmatize(self, text: str) -> List[str]:
        tokens = word_tokenize(text)
        tagged = pos_tag(tokens)
        pcfg   = self.cfg.preprocessing

        allowed = set(self._PRIMARY_POS)
        if pcfg.keep_verbs:
            allowed |= self._SECONDARY_POS

        out = []
        for word, tag in tagged:
            if tag not in allowed:        continue
            if word in self.stopwords:    continue
            if len(word) < pcfg.min_word_len: continue
            if not word.isalpha():        continue
            lemma = self.lemmatizer.lemmatize(word, pos=_penn_to_wn(tag))
            if lemma in self.stopwords or len(lemma) < pcfg.min_word_len:
                continue
            out.append(lemma)
        return out

    def _load_and_tokenize(self, filepath: str) -> List[List[str]]:
        _LOG.info(f"Reading {filepath}")
        with open(filepath, "r", encoding="utf-8", errors="replace") as f:
            raw = [l.strip() for l in f if l.strip()]
        _LOG.info(f"  {len(raw):,} raw lines")

        pcfg = self.cfg.preprocessing
        docs = []
        for i, line in enumerate(raw):
            if i % 5000 == 0 and i > 0:
                _LOG.info(f"  tokenise {i:,}/{len(raw):,}")
            toks = self._tokenize_and_lemmatize(self._normalize(line))
            if len(toks) >= pcfg.min_doc_len:
                docs.append(toks)
        _LOG.info(f"  kept {len(docs):,} docs after POS+length filter")
        return docs

    # ────────────────────────────────────────────────────────────────
    #  step 2 — phrase discovery (NPMI scoring)
    # ────────────────────────────────────────────────────────────────
    def _build_phrases(self, docs: List[List[str]]) -> List[List[str]]:
        pcfg = self.cfg.preprocessing
        _LOG.info("Detecting bigrams (NPMI scoring)...")
        bigram = Phrases(docs,
                         min_count=pcfg.bigram_min_count,
                         threshold=pcfg.bigram_threshold,
                         scoring="npmi",
                         connector_words=ENGLISH_CONNECTOR_WORDS)
        bigram_ph = Phraser(bigram)
        docs_bi = [bigram_ph[d] for d in docs]

        if pcfg.use_trigrams:
            _LOG.info("Detecting trigrams...")
            trigram = Phrases(docs_bi,
                              min_count=pcfg.bigram_min_count,
                              threshold=pcfg.trigram_threshold,
                              scoring="npmi",
                              connector_words=ENGLISH_CONNECTOR_WORDS)
            trigram_ph = Phraser(trigram)
            docs_bi = [trigram_ph[d] for d in docs_bi]

        n_phr = sum(1 for d in docs_bi for w in d if "_" in w)
        _LOG.info(f"  {n_phr:,} phrase tokens created")
        return docs_bi

    # ────────────────────────────────────────────────────────────────
    #  step 3 — semantic-entropy filter  (tighter in v4)
    # ────────────────────────────────────────────────────────────────
    @staticmethod
    def _word_entropy(docs: List[List[str]]) -> Dict[str, float]:
        cooc = defaultdict(Counter)
        for doc in docs:
            uniq = list(set(doc))
            for i, w in enumerate(uniq):
                for j, u in enumerate(uniq):
                    if i != j:
                        cooc[w][u] += 1
        entropy = {}
        for w, nb in cooc.items():
            total = sum(nb.values())
            if total == 0:
                entropy[w] = 0.0
                continue
            probs = np.array(list(nb.values())) / total
            entropy[w] = float(-np.sum(probs * np.log2(probs + 1e-12)))
        return entropy

    def _entropy_filter(self, docs: List[List[str]]) -> List[List[str]]:
        _LOG.info("Computing word-level context entropy ...")
        ent = self._word_entropy(docs)
        if not ent:
            return docs
        vals  = np.array(list(ent.values()))
        # tightened: drop the 12 % most-generic and the 4 % ubiquitous
        floor = np.percentile(vals, 12)
        ceil  = np.percentile(vals, 96)
        keep  = {w for w, e in ent.items() if floor <= e <= ceil}

        filt = [[w for w in d if w in keep] for d in docs]
        filt = [d for d in filt if len(d) >= self.cfg.preprocessing.min_doc_len]
        _LOG.info(f"  entropy-filter dropped {len(ent)-len(keep):,} words; "
                  f"kept {len(filt):,} docs")
        return filt

    # ────────────────────────────────────────────────────────────────
    #  step 4 — build dictionary + BoW  (aggressive filter_extremes)
    # ────────────────────────────────────────────────────────────────
    def _build_dictionary(self, docs: List[List[str]]) -> Dictionary:
        d = Dictionary(docs)
        _LOG.info(f"  dict before filter: {len(d):,}")
        d.filter_extremes(
            no_below=self.cfg.preprocessing.min_word_count,
            no_above=self.cfg.preprocessing.max_word_freq,
            keep_n=self.cfg.preprocessing.max_vocab,
        )
        d.compactify()
        _LOG.info(f"  dict after filter:  {len(d):,}  "
                  f"(no_below={self.cfg.preprocessing.min_word_count}, "
                  f"no_above={self.cfg.preprocessing.max_word_freq}, "
                  f"keep_n={self.cfg.preprocessing.max_vocab})")
        return d

    @staticmethod
    def _docs_to_bow(docs: List[List[str]], d: Dictionary) -> sparse.csr_matrix:
        V, D = len(d), len(docs)
        rows, cols, data = [], [], []
        for i, doc in enumerate(docs):
            for tid, cnt in d.doc2bow(doc):
                rows.append(i); cols.append(tid); data.append(cnt)
        return sparse.csr_matrix((data, (rows, cols)),
                                 shape=(D, V), dtype=np.float32)

    # ────────────────────────────────────────────────────────────────
    #  step 5 — contextual document embeddings (NEW v4)
    # ────────────────────────────────────────────────────────────────
    def _encode_docs(self, docs: List[List[str]], output_dir: str):
        model_name = getattr(self.cfg.dataset, "contextual_model",
                             DEFAULT_MODEL)
        cache = os.path.join(output_dir, "doc_ctx_emb.pt")
        _LOG.info(f"Computing contextual document embeddings with "
                  f"{model_name} (cache -> {cache})")
        return load_contextual_doc_embeddings(
            docs, model_name=model_name, cache_path=cache,
        )

    # ────────────────────────────────────────────────────────────────
    #  master entry point
    # ────────────────────────────────────────────────────────────────
    def run(self, output_dir: str) -> dict:
        """Runs the full pipeline, persists all artifacts, returns a report."""
        from .reference_corpus import build_reference_corpus

        os.makedirs(output_dir, exist_ok=True)
        report = {}

        # step 1
        docs = self._load_and_tokenize(self.cfg.dataset.input_file)
        report["n_docs_after_tokenize"] = len(docs)

        # step 2
        docs = self._build_phrases(docs)
        report["n_docs_after_phrases"] = len(docs)

        # step 3
        docs = self._entropy_filter(docs)
        report["n_docs_after_entropy"] = len(docs)

        # step 4
        dictionary = self._build_dictionary(docs)
        vocab_set  = set(dictionary.token2id.keys())
        docs       = [[w for w in d if w in vocab_set] for d in docs]
        docs       = [d for d in docs if len(d) >= self.cfg.preprocessing.min_doc_len]
        bow        = self._docs_to_bow(docs, dictionary)

        report["vocab_size"]      = len(dictionary)
        report["n_docs_final"]    = len(docs)
        report["avg_doc_length"]  = float(np.mean([len(d) for d in docs]))
        report["sparsity"]        = 1.0 - (bow.nnz / (bow.shape[0] * bow.shape[1]))

        # step 5 — reference corpus for coherence
        ref = build_reference_corpus(docs, self.cfg.preprocessing.ref_window_size)
        report["n_reference_windows"] = len(ref)

        # step 6 — contextual document embeddings (NEW in v4)
        doc_emb = self._encode_docs(docs, output_dir)
        report["ctx_embed_dim"] = int(doc_emb.shape[1])
        report["contextual_model"] = getattr(self.cfg.dataset,
                                             "contextual_model",
                                             DEFAULT_MODEL)

        # persist
        with open(os.path.join(output_dir, "clean_docs.pkl"), "wb") as f:
            pickle.dump(docs, f)
        dictionary.save(os.path.join(output_dir, "dictionary.gensim"))
        sparse.save_npz(os.path.join(output_dir, "bow_matrix.npz"), bow)
        with open(os.path.join(output_dir, "reference_corpus.pkl"), "wb") as f:
            pickle.dump(ref, f)
        with open(os.path.join(output_dir, "preprocessing_report.json"), "w") as f:
            json.dump(report, f, indent=2)

        _LOG.info("=" * 62)
        _LOG.info("PREPROCESSING REPORT")
        for k, v in report.items():
            _LOG.info(f"  {k:28s}: {v}")
        _LOG.info(f"  artifacts -> {output_dir}")
        return report

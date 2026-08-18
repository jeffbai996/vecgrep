"""Nothing stopped the embedder splitting again, and that is the actual bug.

Two corpora on the same backend with different model names is not corruption —
every search still answers — so it hides. Ollama keeps only
OLLAMA_MAX_LOADED_MODELS resident and treats each name as a separate model, so
any search crossing the split evicts and reloads: 900 loads in 6 hours on the
live install (2026-08-18), embeds at 13-28s against a 0.13s warm baseline. It
surfaced as "retrieval feels slow", months after the drift.
"""
from vecgrep.backend.store.corpora import Corpus


def _reg(svc, name, model, backend="ollama"):
    svc.registry.upsert(Corpus(name=name, embed_backend=backend,
                               embed_model=model, dim=8, chunk_count=0))


def test_one_model_everywhere_is_clean(svc):
    _reg(svc, "a", "bge-m3:batch8k")
    _reg(svc, "b", "bge-m3:batch8k")
    kinds = [i["kind"] for i in svc.diagnose()]
    assert "embed_model_split" not in kinds


def test_the_minority_corpus_is_named(svc):
    _reg(svc, "a", "bge-m3:batch8k")
    _reg(svc, "b", "bge-m3:batch8k")
    _reg(svc, "c", "bge-m3")
    split = [i for i in svc.diagnose() if i["kind"] == "embed_model_split"]
    assert [i["corpus"] for i in split] == ["c"]
    assert "bge-m3:batch8k" in split[0]["detail"]
    assert split[0]["fixable"] is False


def test_a_different_backend_is_not_a_split(svc):
    """openai and ollama corpora obviously use different model names; only
    disagreement WITHIN one backend costs anything."""
    _reg(svc, "a", "bge-m3:batch8k", backend="ollama")
    _reg(svc, "b", "text-embedding-3-small", backend="openai")
    assert [i for i in svc.diagnose() if i["kind"] == "embed_model_split"] == []


def test_filtering_still_compares_against_the_whole_install(svc):
    """`--corpora c` must not make c its own majority and report clean."""
    _reg(svc, "a", "bge-m3:batch8k")
    _reg(svc, "b", "bge-m3:batch8k")
    _reg(svc, "c", "bge-m3")
    split = [i for i in svc.diagnose(corpora={"c"}) if i["kind"] == "embed_model_split"]
    assert [i["corpus"] for i in split] == ["c"]

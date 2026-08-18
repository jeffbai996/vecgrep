"""Retrieval-quality eval harness for real corpora.

Every storage or ranking change is a quality/size trade, and this package is
the instrument that prices one: it builds a SIDE-BY-SIDE copy of a live corpus
under a candidate configuration, runs a gold query set against it, and reports
recall@k / precision@k / MRR / latency / bytes-on-disk as one row per
configuration so a change that saves bytes and quietly costs recall is visible
in a single table.

Nothing here touches a live corpus. Builds go to `eval-*` corpora; the gold set
and the report live outside the repo (see `gold.py`).

    python -m vecgrep.eval build  --from chats --to eval-chats-base
    python -m vecgrep.eval run    --gold ~/.vecgrep/eval/gold.json --config ~/.vecgrep/eval/configs.json
    python -m vecgrep.eval report ~/.vecgrep/eval/reports/*.json
"""

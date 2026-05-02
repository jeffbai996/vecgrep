"""Adapter detection + load tests.

Past bugs / risks covered:
- Detection collisions on .json (Claude vs ChatGPT vs random JSON)
- Detection misfires on .jsonl (must look at first line, not just extension)
- ChatGPT linearization following the 'main' thread, not random branches
"""
from __future__ import annotations

import json
from pathlib import Path

from vecgrep.backend.ingestion.adapters import detect_adapter
from vecgrep.backend.ingestion.adapters.discord_jsonl import DiscordJSONLAdapter
from vecgrep.backend.ingestion.adapters.claude_export import ClaudeExportAdapter
from vecgrep.backend.ingestion.adapters.chatgpt_export import ChatGPTExportAdapter
from vecgrep.backend.ingestion.adapters.markdown import MarkdownAdapter
from vecgrep.backend.ingestion.adapters.plaintext import PlaintextAdapter


def test_detects_markdown(tmp_path):
    p = tmp_path / "x.md"
    p.write_text("# hi")
    assert detect_adapter(str(p)) is MarkdownAdapter


def test_detects_plaintext(tmp_path):
    p = tmp_path / "x.txt"
    p.write_text("hi")
    assert detect_adapter(str(p)) is PlaintextAdapter


def test_discord_jsonl_detected_by_content(tmp_path):
    p = tmp_path / "ch.jsonl"
    p.write_text(json.dumps({"author": "alice", "content": "hi"}) + "\n")
    assert detect_adapter(str(p)) is DiscordJSONLAdapter


def test_discord_jsonl_rejects_non_discord_jsonl(tmp_path):
    p = tmp_path / "metrics.jsonl"
    p.write_text(json.dumps({"metric": "cpu", "value": 0.5}) + "\n")
    # No author/content → not discord. Either falls back to plaintext (yes,
    # .jsonl isn't in plaintext exts) or raises AdapterError. Either is fine
    # — the key invariant is we don't misclassify it as discord.
    assert not DiscordJSONLAdapter.matches(str(p))


def test_claude_export_detected(tmp_path):
    p = tmp_path / "claude.json"
    p.write_text(
        json.dumps(
            [
                {
                    "uuid": "x",
                    "name": "test",
                    "created_at": "2026",
                    "chat_messages": [
                        {"sender": "human", "content": "hi"},
                        {"sender": "assistant", "content": "yes"},
                    ],
                }
            ]
        )
    )
    assert detect_adapter(str(p)) is ClaudeExportAdapter


def test_chatgpt_export_detected(tmp_path):
    p = tmp_path / "chatgpt.json"
    p.write_text(
        json.dumps(
            [
                {
                    "id": "c1",
                    "title": "test",
                    "create_time": 0,
                    "mapping": {
                        "n0": {"id": "n0", "parent": None, "children": ["n1"], "message": None},
                        "n1": {
                            "id": "n1",
                            "parent": "n0",
                            "children": [],
                            "message": {
                                "author": {"role": "user"},
                                "content": {"content_type": "text", "parts": ["hi"]},
                            },
                        },
                    },
                }
            ]
        )
    )
    assert detect_adapter(str(p)) is ChatGPTExportAdapter


def test_claude_and_chatgpt_dont_collide(tmp_path):
    """Both use .json — make sure schemas keep them separate."""
    claude = tmp_path / "claude.json"
    claude.write_text(
        json.dumps(
            [{"uuid": "x", "chat_messages": [{"sender": "u", "content": "hi"}]}]
        )
    )
    chatgpt = tmp_path / "chatgpt.json"
    chatgpt.write_text(
        json.dumps(
            [
                {
                    "title": "x",
                    "create_time": 0,
                    "mapping": {
                        "n0": {"id": "n0", "parent": None, "children": [], "message": None}
                    },
                }
            ]
        )
    )
    assert ClaudeExportAdapter.matches(str(claude))
    assert not ClaudeExportAdapter.matches(str(chatgpt))
    assert ChatGPTExportAdapter.matches(str(chatgpt))
    assert not ChatGPTExportAdapter.matches(str(claude))


def test_chatgpt_linearizes_main_thread(tmp_path):
    """When the message graph branches (edits / regenerations), we follow
    the first child. Branch siblings are intentionally dropped — they
    represent alternate threads the user didn't pick.
    """
    p = tmp_path / "chatgpt.json"
    p.write_text(
        json.dumps(
            [
                {
                    "id": "c",
                    "title": "branchy",
                    "create_time": 0,
                    "mapping": {
                        "n0": {"id": "n0", "parent": None, "children": ["n1"], "message": None},
                        "n1": {
                            "id": "n1",
                            "parent": "n0",
                            # Two children = a branch. We take the first.
                            "children": ["n2", "n3"],
                            "message": {
                                "author": {"role": "user"},
                                "content": {"content_type": "text", "parts": ["q?"]},
                            },
                        },
                        "n2": {
                            "id": "n2",
                            "parent": "n1",
                            "children": [],
                            "message": {
                                "author": {"role": "assistant"},
                                "content": {"content_type": "text", "parts": ["main answer"]},
                            },
                        },
                        "n3": {
                            "id": "n3",
                            "parent": "n1",
                            "children": [],
                            "message": {
                                "author": {"role": "assistant"},
                                "content": {"content_type": "text", "parts": ["alt regen"]},
                            },
                        },
                    },
                }
            ]
        )
    )
    docs = list(ChatGPTExportAdapter().load(str(p)))
    assert len(docs) == 1
    assert "main answer" in docs[0].text
    assert "alt regen" not in docs[0].text


def test_discord_load_preserves_message_order(tmp_path):
    p = tmp_path / "ch.jsonl"
    p.write_text(
        "\n".join(
            json.dumps({"author": "alice", "content": f"msg{i}"})
            for i in range(3)
        )
    )
    docs = list(DiscordJSONLAdapter().load(str(p)))
    assert len(docs) == 1
    text = docs[0].text
    assert text.index("msg0") < text.index("msg1") < text.index("msg2")

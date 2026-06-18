"""Write-tool: safe, human-gated, append-only writes into a vecgrep corpus.

Architecture (see the design spec):
- Files on disk are the source of truth; the vector index is a derived view.
- A write is a 4-step pipeline: propose -> confirm -> write -> re-embed+verify.
- Bots/agents may PROPOSE (writes nothing); only a human CONFIRMS (authorizes).
- Append-only: edits write a new version + mark the old superseded; no
  destructive overwrite path exists.
- No read->write loop: ingested content can never become a write proposal.

This package is built in phases. Phase 2 (here) is propose + preview only —
it constructs a Proposal and renders the file it WOULD write, but writes
nothing and mutates no state.
"""
from .proposal import Proposal, ProposalError, propose, render_doc, next_doc_id

__all__ = ["Proposal", "ProposalError", "propose", "render_doc", "next_doc_id"]

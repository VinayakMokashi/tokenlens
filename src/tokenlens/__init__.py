"""tokenlens: analyze Claude Code session transcripts.

Claude Code writes one JSONL transcript per session under
``~/.claude/projects/<encoded-project-dir>/<session-id>.jsonl``. tokenlens
reads those files and answers three questions:

1. What did this session cost, priced per turn at the model that actually
   ran it, including the 5-minute vs 1-hour cache-write tiers?
2. Where did the tokens go: how did the context window grow, which tool
   results were expensive, and when did compaction reset it?
3. What would it have cost on a different model, and which specific turns
   were wasteful?

The command-line interface has no third-party dependencies. The optional
web dashboard requires Flask (``pip install tokenlens[web]``).
"""

__version__ = "0.1.0"

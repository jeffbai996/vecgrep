# The rerank extra installs sentence-transformers; the dev environment does
# not. This stub covers only the cross-encoder vecgrep constructs.
from typing import Any

class CrossEncoder:
    def __init__(self, model_name_or_path: str, **kwargs: Any) -> None: ...
    def __getattr__(self, name: str) -> Any: ...

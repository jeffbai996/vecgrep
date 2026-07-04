# Single source of truth for the vecgrep version. pyproject.toml reads this
# dynamically (see [tool.setuptools.dynamic]), and the CLI / FastAPI / health
# endpoint import it at runtime — so this line is the ONLY place to bump.
__version__ = "0.7.0"

"""Check a base wheel in a clean environment without optional extras or inference."""
from __future__ import annotations

import importlib.util
import os
import tempfile
from importlib.resources import files
from pathlib import Path


def main() -> None:
    # Run with -I so an editable checkout or PYTHONPATH cannot mask a broken wheel.
    for module in ("PIL", "pypdfium2", "mcp", "tree_sitter", "tree_sitter_typescript",
                   "openai", "sentence_transformers"):
        assert importlib.util.find_spec(module) is None, f"Unexpected optional dependency: {module}"

    with tempfile.TemporaryDirectory(prefix="vecgrep-base-smoke-") as directory:
        for key in list(os.environ):
            if key.startswith("VECGREP_"):
                del os.environ[key]
        os.environ["VECGREP_HOME"] = str(Path(directory) / "home")

        from pypdf import PdfWriter
        from pypdf.generic import DecodedStreamObject, DictionaryObject, NameObject

        from vecgrep.backend.ingestion.adapters import detect_adapter
        from vecgrep.backend.ingestion.adapters.image import ImageAdapter
        from vecgrep.backend.ingestion.adapters.pdf import PDFAdapter
        from vecgrep.backend.main import app

        assert app is not None
        frontend = files("vecgrep").joinpath("frontend", "dist")
        assert frontend.joinpath("index.html").is_file()
        assets = list(frontend.joinpath("assets").iterdir())
        assert any(asset.name.endswith(".js") for asset in assets)
        assert any(asset.name.endswith(".css") for asset in assets)
        text = Path(directory) / "notes.txt"
        text.write_text("Native text remains available.", encoding="utf-8")
        assert next(detect_adapter(str(text))().load(str(text))).text == text.read_text()
        image = Path(directory) / "scan.png"
        image.write_bytes(b"visual ingestion is disabled by default")
        assert not ImageAdapter.matches(str(image))

        # Native PDF text and page anchors must work without the visual extra.
        writer = PdfWriter()
        font = DictionaryObject({
            NameObject("/Type"): NameObject("/Font"),
            NameObject("/Subtype"): NameObject("/Type1"),
            NameObject("/BaseFont"): NameObject("/Helvetica"),
        })
        for number in (1, 2):
            page = writer.add_blank_page(width=200, height=100)
            page[NameObject("/Resources")] = DictionaryObject({
                NameObject("/Font"): DictionaryObject({NameObject("/F1"): font}),
            })
            content = DecodedStreamObject()
            content.set_data(f"BT /F1 12 Tf 10 50 Td (Native page {number}) Tj ET".encode())
            page[NameObject("/Contents")] = content
        pdf = Path(directory) / "native.pdf"
        writer.write(pdf)
        document = next(PDFAdapter().load(str(pdf)))
        assert document.text == "[page 1]\nNative page 1\n\n[page 2]\nNative page 2"
        assert document.metadata["pages"] == 2
        assert "visual_state" not in document.metadata


if __name__ == "__main__":
    main()

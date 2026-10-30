"""Browser-form helpers for OAuth tests; all traffic stays in the ASGI client."""

from __future__ import annotations

from html.parser import HTMLParser


class FormFields(HTMLParser):
    def __init__(self, text: str) -> None:
        super().__init__()
        self.values: dict[str, str] = {}
        self.feed(text)

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        values = dict(attrs)
        if tag == "input" and values.get("type") == "hidden":
            name = values.get("name")
            if name is not None:
                val = values.get("value")
                self.values[name] = val if val is not None else ""


def submit(client, *, data: dict, headers: dict | None = None, **kwargs):
    page = client.get(
        "/oauth/unlock",
        params={"next": data.get("next", "/authorize?client_id=x")},
        headers=headers,
    )
    assert page.status_code == 200, page.text
    fields = FormFields(page.text).values
    return client.post(
        "/oauth/unlock", data={**fields, **data}, headers=headers, **kwargs
    )

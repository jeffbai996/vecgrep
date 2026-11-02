"""Self-contained presentation for script-free OAuth approval pages."""
from base64 import b64encode
from functools import lru_cache
from html import escape
from pathlib import Path


@lru_cache(maxsize=1)
def theme_css() -> str:
    """Embed bundled fonts so approval documents make no third-party requests."""
    fonts = []
    for name, filename in (("Plus Jakarta Sans", "jakarta.woff2"),
                           ("DM Sans", "dm-sans.woff2"),
                           ("Anthropic Sans", "AnthropicSansVariable-TextRegular.woff2")):
        data = b64encode((Path(__file__).with_name("oauth_fonts") / filename).read_bytes()).decode("ascii")
        fonts.append('@font-face{font-family:"' + name + '";src:url(data:font/woff2;base64,' + data
                     + ') format("woff2");font-weight:100 900;font-style:normal;font-display:swap;}')
    return "\n".join(fonts) + '''
:root{--font:"Anthropic Sans",system-ui,sans-serif;--ui:"DM Sans","Anthropic Sans",system-ui,sans-serif;--heading:"Plus Jakarta Sans","DM Sans",system-ui,sans-serif;--bg:#090a0c;--bg-2:#111317;--bg-3:#191c21;--fg:#edf0f4;--fg-2:#b3bac4;--muted:#8b95a2;--line:#2a3039;--accent:#a78bfa;--accent-hi:#c4b5fd;--accent-wash:#a78bfa0d}
*{box-sizing:border-box}html{color-scheme:dark}body{margin:0;min-height:100dvh;display:grid;place-items:center;padding:32px 20px;background:var(--bg);color:var(--fg);font-family:var(--font);-webkit-font-smoothing:antialiased}body::before{background:radial-gradient(ellipse at 50% 0%,#a78bfa08,transparent 65%)}
main{width:100%;max-width:460px;position:relative}.brand{font-family:var(--ui);font-size:12px;letter-spacing:0;color:var(--muted);margin:0 4px 18px;display:flex;align-items:center;gap:10px;flex-wrap:wrap;text-transform:none}.brand b{font-family:var(--heading);font-size:14px;font-weight:650;letter-spacing:-.02em;color:var(--fg)}.brand .host{font-size:11px;overflow-wrap:anywhere}.brand .mark{background:var(--bg-3);border:1px solid var(--line);border-radius:8px;width:28px;height:28px;display:grid;place-items:center;color:var(--accent)}
.card{border:1px solid var(--line);border-radius:18px;padding:30px;background:var(--bg-2);box-shadow:0 16px 60px #0003}h1{font:650 26px/1.2 var(--heading);letter-spacing:-.055em;margin:0 0 12px;overflow-wrap:anywhere}p.sub{font:400 14px/1.6 var(--font);color:var(--fg-2);margin:0 0 22px;overflow-wrap:anywhere}
ul.scopes{background:var(--bg-3);border:1px solid var(--line);border-radius:12px;padding:16px;margin:0 0 22px;display:grid;gap:14px;list-style:none}ul.scopes li{font-family:var(--ui);display:flex;align-items:flex-start;gap:12px}ul.scopes b{font:600 14px/1.4 var(--ui);display:block}ul.scopes small{font:400 12px/1.5 var(--font);color:var(--muted);display:block;margin-top:3px}ul.scopes .dot{width:6px;height:6px;flex:none;margin-top:7px;border-radius:50%;background:var(--accent);box-shadow:none}
label{font:550 12px/1.4 var(--ui);letter-spacing:0;text-transform:none;margin-bottom:8px}input[type=password]{font:400 16px var(--ui);width:100%;padding:13px 14px;border:1px solid var(--line);border-radius:10px;color:var(--fg);background:var(--bg-3);letter-spacing:.04em}button,.continue{font:600 14px/1.4 var(--ui);width:100%;padding:13px 18px;border-radius:10px;background:var(--accent);color:#1c1033;border:1px solid transparent;text-align:center;text-decoration:none;display:block;cursor:pointer;margin-top:14px}button:hover,.continue:hover{background:var(--accent-hi)}button:focus-visible,.continue:focus-visible,input:focus-visible{outline:2px solid var(--accent);outline-offset:4px}
.details{font:400 12px/1.5 var(--ui);margin:0 0 22px}.details dt{color:var(--muted);margin-top:10px}.details dd{font-family:var(--font);color:var(--fg-2);margin:3px 0 0;overflow-wrap:anywhere}footer{font:400 11px/1.5 var(--ui);color:var(--muted);display:flex;justify-content:space-between;gap:12px;flex-wrap:wrap;margin:18px 4px 0;letter-spacing:0}.approved-mark{width:44px;height:44px;border-radius:50%;border:1px solid #89d6ae55;background:#89d6ae0c;color:#a0dbba;display:grid;place-items:center;margin-bottom:22px;font:500 22px var(--ui)}
@media(max-width:480px){body{padding:24px 16px}.card{padding:24px}h1{font-size:23px}}
'''


def render_navigation(destination: str) -> str:
    """Preserve automatic navigation, with a usable fallback when it is blocked."""
    target = escape(destination, quote=True)
    return ('<!doctype html><html lang="en"><head><meta charset="utf-8">'
            '<meta name="viewport" content="width=device-width, initial-scale=1">'
            '<meta name="color-scheme" content="dark"><title>Connection approved</title>'
            f'<meta http-equiv="refresh" content="0;url={target}">'
            '<style>' + theme_css() + '</style></head><body><main>'
            '<div class="brand">Authorization</div><section class="card">'
            '<div class="approved-mark" aria-hidden="true">✓</div>'
            '<h1>Connection approved</h1><p class="sub">Returning to the application. '
            'Continue to finish connecting if this page stays open.</p>'
            f'<a href="{target}" class="continue">Continue connection</a>'
            '</section></main></body></html>')

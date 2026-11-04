"""
Contenuti della home di un sito della chat: testo descrittivo e pulsanti.

Il testo finisce nell'editor del sito di amministrazione, che è una pagina
autenticata: da lì non deve poter partire alcuno script, chiunque l'abbia scritto
(file caricati da riga di comando, traduzione del modello). Per questo si salva
solo ciò che è in elenco.
"""

import html
import json
from html.parser import HTMLParser
from urllib.parse import urlparse

ALLOWED_TAGS = {"p", "br", "hr", "h1", "h2", "h3", "b", "strong", "i", "em", "u", "ul", "ol", "li", "a",
                "blockquote", "div", "span"}
VOID_TAGS = {"br", "hr"}
DROPPED_WITH_CONTENT = {"script", "style", "iframe", "object", "embed", "svg", "math", "template", "noscript"}
MAX_BUTTONS = 20


def _safe_link(url, schemes):
    url = (url or "").strip()
    return url if urlparse(url).scheme.lower() in schemes else None


class _Cleaner(HTMLParser):
    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.out = []
        self.has_text = False
        self.dropping = 0

    def handle_starttag(self, tag, attrs):
        if tag in DROPPED_WITH_CONTENT:
            self.dropping += 1
        elif not self.dropping and tag in ALLOWED_TAGS:
            href = _safe_link(dict(attrs).get("href"), {"http", "https", "mailto"}) if tag == "a" else None
            self.out.append(f'<a href="{html.escape(href)}">' if href else f"<{tag}>")

    def handle_endtag(self, tag):
        if tag in DROPPED_WITH_CONTENT:
            self.dropping = max(0, self.dropping - 1)
        elif not self.dropping and tag in ALLOWED_TAGS and tag not in VOID_TAGS:
            self.out.append(f"</{tag}>")

    def handle_data(self, data):
        if not self.dropping:
            self.has_text = self.has_text or bool(data.strip())
            self.out.append(html.escape(data, quote=False))


def clean_html(value):
    """L'HTML con i soli tag ammessi e senza attributi, tranne i link; None se non resta testo."""
    if not value:
        return None
    cleaner = _Cleaner()
    cleaner.feed(value)
    cleaner.close()
    return "".join(cleaner.out).strip() if cleaner.has_text else None


def clean_buttons(value):
    """Lista di pulsanti {label_it, label_en, url} validata. ValueError con il motivo, se non va bene."""
    if value is None:
        return []
    if not isinstance(value, list) or len(value) > MAX_BUTTONS:
        raise ValueError(f"I pulsanti devono essere una lista di al massimo {MAX_BUTTONS} voci")
    buttons = []
    for item in value:
        item = item if isinstance(item, dict) else {}
        label_it = str(item.get("label_it") or "").strip()
        label_en = str(item.get("label_en") or "").strip()
        url = _safe_link(str(item.get("url") or ""), {"http", "https"})
        if not label_it or len(label_it) > 100 or len(label_en) > 100:
            raise ValueError("Ogni pulsante deve avere un'etichetta italiana, di al massimo 100 caratteri")
        if not url or len(url) > 2000:
            raise ValueError(f"Il pulsante '{label_it}' deve avere un indirizzo http:// o https://")
        buttons.append({"label_it": label_it, "label_en": label_en or None, "url": url})
    return buttons


def buttons_from_db(value):
    return json.loads(value) if value else []

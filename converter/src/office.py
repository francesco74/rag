"""
Documenti d'ufficio in Markdown: testi (docx, odt) con pandoc, fogli di calcolo
(xlsx, ods) con pandas, un foglio per sezione. Non c'è OCR: le immagini dentro
i documenti si ignorano.
"""

import math
from datetime import date, datetime, time
from pathlib import Path

TEXT_DOCUMENTS = {".docx", ".odt"}
SPREADSHEETS = {".xlsx", ".ods"}
OFFICE_EXTENSIONS = TEXT_DOCUMENTS | SPREADSHEETS


def document_to_markdown(path: Path) -> str:
    import pypandoc
    # Senza HTML: le tabelle che il Markdown non sa rappresentare diventano testo semplice.
    return pypandoc.convert_file(str(path), "gfm-raw_html", format=path.suffix[1:],
                                 extra_args=["--wrap=none"])


def _cell(value) -> str:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return ""
    if isinstance(value, datetime):
        return value.strftime("%Y-%m-%d") if value.time() == time() else value.strftime("%Y-%m-%d %H:%M")
    if isinstance(value, date):
        return value.strftime("%Y-%m-%d")
    if isinstance(value, float) and value.is_integer():
        return str(int(value))
    # "|" spezzerebbe la tabella, l'a capo la riga.
    return str(value).replace("|", "\\|").replace("\r", " ").replace("\n", " ").strip()


def spreadsheet_to_markdown(path: Path) -> str:
    import pandas as pd
    engine = "odf" if path.suffix.lower() == ".ods" else "openpyxl"
    sheets = pd.read_excel(path, sheet_name=None, header=None, engine=engine)
    parts = []
    for name, frame in sheets.items():
        rows = [[_cell(v) for v in row] for row in frame.itertuples(index=False)]
        rows = [r for r in rows if any(r)]
        if not rows:
            continue
        used = [i for i in range(len(rows[0])) if any(r[i] for r in rows)]
        rows = [[r[i] for i in used] for r in rows]
        # La prima riga piena fa da intestazione, come di solito nei fogli d'ufficio.
        header, body = rows[0], rows[1:]
        lines = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
        lines += ["| " + " | ".join(r) + " |" for r in body]
        parts.append(f"## {name}\n\n" + "\n".join(lines))
    return "\n\n".join(parts)


def office_to_markdown(path: Path) -> str:
    return spreadsheet_to_markdown(path) if path.suffix.lower() in SPREADSHEETS else document_to_markdown(path)

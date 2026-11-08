"""Controlli sui file in ingresso, comuni a direct.py e al caricamento dall'amministrazione.

Le funzioni sono quelle di pa-common (pa_common.ingresso, condiviso con atti-pa); qui resta
la politica del RAG: quali formati si accettano.
"""

from pa_common.ingresso import nome_sbustato, sbusta_p7m, tipo_reale

# Estensioni ammesse per il contenuto FINALE (dopo eventuale sbustamento p7m).
# Documenti d'ufficio: testo e fogli di calcolo, in formato Microsoft e OpenDocument.
OFFICE_EXTENSIONS = {".docx", ".odt", ".xlsx", ".ods"}
SUPPORTED_EXTENSIONS = {".pdf", ".txt", ".md", ".png", ".jpg", ".jpeg"} | OFFICE_EXTENSIONS
# Estensioni accettate in INGRESSO: le stesse ammesse, più ".p7m" (buste di
# firma digitale CAdES/PKCS7, sbustate prima di verificare il tipo reale).
INPUT_EXTENSIONS = SUPPORTED_EXTENSIONS | {".p7m"}


def detect_extension_from_content(content: bytes) -> str | None:
    """Estensione reale dal contenuto (firma iniziale; per gli zip, cosa contengono), None se
    non è un formato con una firma riconoscibile. Può riconoscere anche formati che il RAG non
    ammette (es. .doc): il chiamante li confronta con SUPPORTED_EXTENSIONS."""
    return tipo_reale(content)


def try_extract_from_pkcs7(file_bytes: bytes, filename: str = "") -> bytes | None:
    """Il documento dentro una busta di firma PKCS7/CAdES (anche in base64), None se non è una
    busta. Solleva ValueError (BustaNonValida) se è una busta ma il documento non si estrae."""
    return sbusta_p7m(file_bytes)


def unwrapped_name(p7m_name: str, detected_ext: str) -> str:
    """"fattura.pdf.p7m" -> "fattura.pdf"; "fattura.p7m" -> "fattura.pdf"."""
    return nome_sbustato(p7m_name, detected_ext)

"""Controlli sui file in ingresso, comuni a direct.py e al caricamento dall'amministrazione."""

import io
import logging
import zipfile

log = logging.getLogger("INTAKE")

# Estensioni ammesse per il contenuto FINALE (dopo eventuale sbustamento p7m).
# Documenti d'ufficio: testo e fogli di calcolo, in formato Microsoft e OpenDocument.
OFFICE_EXTENSIONS = {".docx", ".odt", ".xlsx", ".ods"}
SUPPORTED_EXTENSIONS = {".pdf", ".txt", ".md", ".png", ".jpg", ".jpeg"} | OFFICE_EXTENSIONS
# Estensioni accettate in INGRESSO: le stesse ammesse, più ".p7m" (buste di
# firma digitale CAdES/PKCS7, sbustate prima di verificare il tipo reale).
INPUT_EXTENSIONS = SUPPORTED_EXTENSIONS | {".p7m"}

# Magic bytes dei formati supportati, per rilevare il tipo reale del
# contenuto DOPO lo sbustamento (un file .p7m non dice nulla sul contenuto
# che incapsula finché non lo si apre).
_MAGIC_BYTES = (
    (b"%PDF-", ".pdf"),
    (b"\x89PNG\r\n\x1a\n", ".png"),
    (b"\xff\xd8\xff", ".jpg"),
)
# OpenDocument dichiara il tipo nel file "mimetype" dell'archivio zip.
_ODF_TYPES = {"application/vnd.oasis.opendocument.text": ".odt",
              "application/vnd.oasis.opendocument.spreadsheet": ".ods"}


def _office_type(content: bytes) -> str | None:
    """docx, xlsx, odt e ods sono archivi zip: il tipo si riconosce da cosa contengono."""
    try:
        with zipfile.ZipFile(io.BytesIO(content)) as archive:
            names = set(archive.namelist())
            if "mimetype" in names:
                return _ODF_TYPES.get(archive.read("mimetype").decode("ascii", "ignore").strip())
            if "word/document.xml" in names:
                return ".docx"
            if "xl/workbook.xml" in names:
                return ".xlsx"
    except (zipfile.BadZipFile, KeyError, OSError):
        return None
    return None


def detect_extension_from_content(content: bytes) -> str | None:
    """
    Rileva l'estensione reale in base ai magic bytes iniziali e, per gli archivi
    zip, al loro contenuto (docx, xlsx, odt, ods: serve il file intero). Restituisce
    None se il contenuto non corrisponde a nessuno dei formati binari
    riconosciuti — i formati testuali (.txt/.md) non hanno
    una firma binaria affidabile e in pratica non ha senso attendersi che
    una busta di firma digitale li incapsuli, quindi non vengono rilevati
    qui: un simile contenuto verrebbe scartato come "non supportato".
    """
    for magic, ext in _MAGIC_BYTES:
        if content.startswith(magic):
            return ext
    if content.startswith(b"PK\x03\x04"):
        return _office_type(content)
    return None


def try_extract_from_pkcs7(file_bytes: bytes, filename: str = "") -> bytes | None:
    """
    Tenta di interpretare file_bytes come busta di firma PKCS7/CMS (CAdES) e,
    se lo è, ne estrae il contenuto incapsulato.

    Restituisce:
    - None se file_bytes NON è una struttura PKCS7 valida — il chiamante
      deve trattarlo come anomalo per un file dichiarato ".p7m" (a
      differenza di converter.py, dove None per un file .pdf significa
      semplicemente "PDF nativo già in chiaro", qui l'estensione .p7m
      dichiara esplicitamente che ci si aspettava una busta).
    - i bytes del contenuto estratto, se file_bytes è una busta PKCS7 di
      tipo signed_data e l'estrazione riesce.

    Solleva ValueError se la struttura è riconosciuta come PKCS7 signed_data
    ma l'estrazione del contenuto incapsulato fallisce.
    """
    from asn1crypto.cms import ContentInfo

    try:
        content_info = ContentInfo.load(file_bytes)
    except Exception:
        return None

    if content_info['content_type'].native != 'signed_data':
        return None

    try:
        signed_data = content_info['content']
        encap_content_info = signed_data['encap_content_info']
        raw_content = encap_content_info['content'].native
        extracted_bytes = raw_content if isinstance(raw_content, bytes) else encap_content_info['content'].chosen.contents
        log.debug("Sbustamento P7M completato per %s", filename)
        return extracted_bytes
    except Exception as e:
        log.error("Fallimento sbustamento P7M per %s: %s", filename, str(e))
        raise ValueError(f"Decodifica P7M fallita: {str(e)}")


def unwrapped_name(p7m_name: str, detected_ext: str) -> str:
    """"fattura.pdf.p7m" -> "fattura.pdf"; "fattura.p7m" -> "fattura.pdf"."""
    stripped = p7m_name[:-len(".p7m")]
    return stripped if stripped.lower().endswith(detected_ext) else f"{stripped}{detected_ext}"

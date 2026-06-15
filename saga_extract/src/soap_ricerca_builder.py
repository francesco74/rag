from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple, Union, Any, Dict, List
from urllib.parse import urlencode, urlparse, urlunparse, parse_qsl
from xml.sax.saxutils import escape
import xml.etree.ElementTree as ET
import html

import logging
import requests
import sys

# CORRETTO: Aggiunto DeterminaFilter negli import da ricerca_filtri
from ricerca_filtri import DeterminaFilter, DatiUtenteFilter, MetadataItem, RicercaFiltri

# TOML: Python 3.11+ ha tomllib; fallback per versioni precedenti

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        #logging.FileHandler("app.log"), # Saves to a file
        logging.StreamHandler()         # Prints to console
    ]
)
log = logging.getLogger(__name__)

# -----------------------------
# SOAP builder for DocWSRicerche
# -----------------------------

class SoapDocWSRicercheBuilder:
    SOAPENV_NS = "http://schemas.xmlsoap.org/soap/envelope/"
    TEMPURI_NS = "http://tempuri.org/"

    def __init__(self, ricerca_filtri: RicercaFiltri):
        self.ricerca_filtri = ricerca_filtri

    def to_ricerca_filtri_xml(self, pretty: bool = False) -> str:
        return self.ricerca_filtri.to_xml(pretty=pretty)

    def to_soap_envelope(
        self,
        codice_amministrazione: str = "",
        codice_aoo: str = "",
        pretty: bool = True
    ) -> str:
        """
        Genera una richiesta SOAP completa per RicercaDocumentiString.
        RicercaFiltriStr viene inserito come CDATA (non escapato), coerente con l'esempio nel PDF.
        """
        rf_xml = self.to_ricerca_filtri_xml(pretty=False)

        # Escape solo per i campi “normali” (non CDATA)
        ca = escape(codice_amministrazione or "")
        aoo = escape(codice_aoo or "")

        soap = f"""<?xml version="1.0" encoding="utf-8"?>
<soapenv:Envelope xmlns:soapenv="{self.SOAPENV_NS}" xmlns:tem="{self.TEMPURI_NS}">
  <soapenv:Header/>
  <soapenv:Body>
    <tem:RicercaDocumentiString>
      <tem:RicercaFiltriStr><![CDATA[{rf_xml}]]></tem:RicercaFiltriStr>
      <tem:CodiceAmministrazione>{ca}</tem:CodiceAmministrazione>
      <tem:CodiceA00>{aoo}</tem:CodiceA00>
    </tem:RicercaDocumentiString>
  </soapenv:Body>
</soapenv:Envelope>
"""
        if not pretty:
            soap = soap.replace("\n", "").replace("  ", "")
        return soap


# -----------------------------
# HTTP Client for DocWSRicerche
# -----------------------------

def _add_query_params(url: str, params: dict) -> str:
    u = urlparse(url)
    q = dict(parse_qsl(u.query))
    for k, v in params.items():
        if v is None:
            continue
        q[k] = str(v)
    new_query = urlencode(q)
    return urlunparse((u.scheme, u.netloc, u.path, u.params, new_query, u.fragment))


@dataclass
class SoapSendResult:
    http_status: int
    response_text: str
    result_text: Optional[str] = None
    result_xml_root: Optional[ET.Element] = None


class DocWSRicercheClient:
    def __init__(
        self,
        endpoint_url: str,
        cid: Optional[str] = None,
        timeout: Union[int, float] = 30,
        verify_tls: bool = True,
    ):
        self.endpoint_url = endpoint_url
        self.cid = cid
        self.timeout = timeout
        self.verify_tls = verify_tls

    def send_ricerca_documenti_string(self, soap_xml: str, soap_version: str = "1.1") -> SoapSendResult:
        url = self.endpoint_url
        if self.cid:
            url = _add_query_params(url, {"CID": self.cid})

        headers: Dict[str, str] = {}
        if soap_version == "1.2":
            headers["Content-Type"] = "application/soap+xml; charset=utf-8"
        else:
            headers["Content-Type"] = "text/xml; charset=utf-8"
            headers["SOAPAction"] = '"http://tempuri.org/RicercaDocumentiString"'

        log.debug("Avvio chiamata HTTP a: %s", url)
        log.debug("Headers: %s", headers)
        
        resp = requests.post(
            url,
            data=soap_xml,
            headers=headers,
            timeout=self.timeout,
            verify=self.verify_tls,
        )

        # TRACCIAMENTO RISPOSTA SERVER
        log.info("HTTP Status: %s", resp.status_code)
        log.debug("Risposta Server (primi 1500 caratteri): %s", html.unescape(resp.text)[:1500])
    
        if not resp.ok:
            raise RuntimeError(
                f"HTTP {resp.status_code} calling DocWSRicerche. "
                f"Response body (first 1000 chars): {resp.text[:1000]}"
            )

        result_text, result_xml_root = self._extract_result(resp.text)
        return SoapSendResult(
            http_status=resp.status_code,
            response_text=resp.text,
            result_text=result_text,
            result_xml_root=result_xml_root,
        )

    @staticmethod
    def _extract_result(soap_response_xml: str) -> Tuple[Optional[str], Optional[ET.Element]]:
        try:
            root = ET.fromstring(soap_response_xml)
        except ET.ParseError:
            return None, None

        def endswith(tag: str, suffix: str) -> bool:
            return tag == suffix or tag.endswith("}" + suffix)

        result_el = None
        for el in root.iter():
            if endswith(el.tag, "RicercaDocumentiStringResult"):
                result_el = el
                break

        if result_el is None:
            return None, None

        result_text = (result_el.text or "").strip()
        if not result_text:
            return "", None

        try:
            inner_root = ET.fromstring(result_text)
            return result_text, inner_root
        except ET.ParseError:
            return result_text, None


# -----------------------------
# TOML -> RicercaFiltri
# -----------------------------

def _as_str(x: Any) -> Optional[str]:
    if x is None:
        return None
    s = str(x).strip()
    return s if s else None



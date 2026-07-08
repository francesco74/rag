from __future__ import annotations

import base64
import logging
import xml.etree.ElementTree as ET
from typing import Tuple, Dict, Any, Iterator
import requests
import re

from common.config import settings

log = logging.getLogger(__name__)

class WSAttiSoapClient:
    """Client per il Web Service WSAttiSoap di Sicr@Web Maggioli."""
    
    SOAPENV_NS = "http://schemas.xmlsoap.org/soap/envelope/"
    TEM_NS = "http://tempuri.org/"

    def __init__(self, endpoint_url: str, username: str, timeout: int = 300, verify_tls: bool = True):
        self.endpoint_url = endpoint_url
        self.username = username  # La tua logica originale
        self.timeout = timeout
        self.verify_tls = verify_tls
        self.session = requests.Session()

    def __enter__(self) -> WSAttiSoapClient:
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

    def close(self):
        self.session.close()

    def _soap_envelope_leggi_atto_plus(self, filtro_atto_xml: str) -> str:
        # RIPRISTINATO IL TUO PAYLOAD FUNZIONANTE
        return f"""<?xml version="1.0" encoding="utf-8"?>
<soapenv:Envelope xmlns:soapenv="{self.SOAPENV_NS}" xmlns:tem="{self.TEM_NS}">
  <soapenv:Header/>
  <soapenv:Body>
    <tem:LeggiAttoPlus>
      <tem:FiltroAtto><![CDATA[{filtro_atto_xml}]]></tem:FiltroAtto>
      <tem:CodiceAmministrazione>{self.username}</tem:CodiceAmministrazione>
      <tem:CodiceA00></tem:CodiceA00>
    </tem:LeggiAttoPlus>
  </soapenv:Body>
</soapenv:Envelope>
"""

    def _post_soap(self, soap_xml: str) -> str:
        resp = self.session.post(
            self.endpoint_url,
            data=soap_xml.encode("utf-8"),
            headers={"content-type": "text/xml; charset=utf-8", "SOAPAction": "http://tempuri.org/LeggiAttoPlus"},
            timeout=self.timeout,
            verify=self.verify_tls,
        )
        if not resp.ok:
            raise RuntimeError(f"HTTP {resp.status_code}: {resp.text[:1000]}")
        return resp.text

    def leggi_atto_plus(self, id_documento: str) -> Tuple[Iterator[Tuple[bytes, str]], Dict[str, Any]]:
        filtro_xml = f"""<FiltroAttoIn>
  <IdDocumento>{id_documento}</IdDocumento>
  <DownloadAllegati>S</DownloadAllegati>
  <DownloadAllegatiSenzaFirma>N</DownloadAllegatiSenzaFirma>
</FiltroAttoIn>"""

        soap = self._soap_envelope_leggi_atto_plus(filtro_xml)
        log.debug("Dati SOAP per LeggiAttoPlus in invio per UID %s", id_documento)
        resp_xml = self._post_soap(soap)

        tree = ET.fromstring(resp_xml)
        result_el = None
        for el in tree.iter():
            if el.tag.endswith("LeggiAttoPlusResult"):
                result_el = el
                break

        if result_el is None or not (result_el.text or "").strip():
            raise RuntimeError("Tag LeggiAttoPlusResult vuoto o non trovato nella risposta SOAP.")

        clean_inner = re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]', '', result_el.text.strip())

        try:
            inner_tree = ET.fromstring(clean_inner)
        except ET.ParseError as e:
            log.error("XML interno malformato. Porzione: %s", clean_inner[max(0, e.position[1]-50) : e.position[1]+50])
            raise RuntimeError(f"WSAttiSoap Errore Parsing XML interno: {str(e)}")

        errore = inner_tree.findtext(".//Errore")
        if errore and errore.strip():
            raise RuntimeError(f"WSAttiSoap Errore Applicativo: {errore.strip()}")

        oggetto = inner_tree.findtext(".//Oggetto") or ""
        numero_atto = ""
        data_atto = ""
        anno_atto = ""

        atto_el = inner_tree.find(".//Determina") or inner_tree.find(".//Delibera")
        if atto_el is not None:
            numero_atto = atto_el.findtext(".//Numero") or ""
            data_atto = atto_el.findtext(".//Data") or ""
            # Anno vive direttamente sotto Determina/Delibera (es. <Anno>2026</Anno>),
            # non dentro Workflow/Attributi. E' il segnale affidabile per l'anno
            # dell'atto, da preferire al parsing del registro nella risposta di
            # ricerca (RicercaDocumentiString), che per alcuni atti reali può
            # mancare del tutto (vedi decreto presidenziale 1/2024, UID 2119188).
            anno_atto = atto_el.findtext(".//Anno") or ""

        # id_tipo_iter: CONFERMATO dal cliente essere il campo che distingue
        # realmente il sotto-tipo dell'atto (deliberativo vs presidenziale),
        # a differenza del <Tipo>DEC</Tipo> nella ricerca (che non distingue
        # affatto) e del registro Verbale nella risposta di ricerca (che può
        # mancare del tutto per un atto reale, vedi UID 2119188). Valori
        # confermati: 8 = decreto del Presidente, 9 o 19 = decreto
        # deliberativo. Vive dentro Determina/Workflow/Attributi/Attributo
        # con Nome="id_tipo_iter", non a livello Determina/Delibera diretto,
        # quindi va cercato separatamente in tutto l'albero.
        id_tipo_iter = ""
        for attributo_el in inner_tree.findall(".//Attributi/Attributo"):
            nome_attr = (attributo_el.findtext("Nome") or "").strip()
            if nome_attr == "id_tipo_iter":
                id_tipo_iter = (attributo_el.findtext("Valore") or "").strip()
                break

        allegati_el = inner_tree.findall(".//Allegati/Allegato")
        if not allegati_el:
            raise RuntimeError(f"Nessun allegato binario associato all'atto UID {id_documento}")

        attributes = {
            "oggetto": oggetto,
            "numero_atto": numero_atto,
            "data_atto": data_atto,
            "anno_atto": anno_atto,
            "id_tipo_iter": id_tipo_iter,
        }

        def estrai_allegati() -> Iterator[Tuple[bytes, str]]:
            for idx, allg in enumerate(allegati_el):
                file_name = allg.findtext(".//NomeAllegato")
                file_name = file_name.strip() if file_name else f"allegato_{idx}.bin"

                b64_image = allg.findtext(".//Image")
                if not b64_image or not b64_image.strip():
                    log.warning("Payload <Image> vuoto per '%s' (UID %s). Saltato.", file_name, id_documento)
                    continue

                try:
                    yield base64.b64decode(b64_image.strip()), file_name
                except Exception as e:
                    log.error("Errore decodifica Base64 per allegato '%s': %s", file_name, str(e))

        return estrai_allegati(), attributes

def build_client_from_env() -> WSAttiSoapClient:
    return WSAttiSoapClient(
        endpoint_url=settings.docws_atti_endpoint,
        username=settings.ws_username,
        timeout=settings.http_timeout_seconds,
        verify_tls=settings.verify_tls,
    )
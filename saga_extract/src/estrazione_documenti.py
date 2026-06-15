from __future__ import annotations

import base64
import html
import logging
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from typing import Tuple, Dict, Any, List
import requests

from config import settings

log = logging.getLogger(__name__)

@dataclass
class WSAttiCredentials:
    username: str
    password: str

class WSAttiSoapClient:
    """Client per il Web Service WSAttiSoap di Sicr@Web Maggioli."""
    
    SOAPENV_NS = "http://schemas.xmlsoap.org/soap/envelope/"
    TEM_NS = "http://tempuri.org/"

    def __init__(self, endpoint_url: str, credentials: WSAttiCredentials, timeout: int = 30, verify_tls: bool = True):
        self.endpoint_url = endpoint_url
        self.credentials = credentials
        self.timeout = timeout
        self.verify_tls = verify_tls
        self.session = requests.Session()

    def _soap_envelope_leggi_atto_plus(self, filtro_atto_xml: str) -> str:
        # Il tag deve essere rigorosamente CodiceA00 con gli zeri per specifiche Maggioli
        return f"""<?xml version="1.0" encoding="utf-8"?>
<soapenv:Envelope xmlns:soapenv="{self.SOAPENV_NS}" xmlns:tem="{self.TEM_NS}">
  <soapenv:Header/>
  <soapenv:Body>
    <tem:LeggiAttoPlus>
      <tem:FiltroAtto><![CDATA[{filtro_atto_xml}]]></tem:FiltroAtto>
      <tem:CodiceAmministrazione>{self.credentials.username}</tem:CodiceAmministrazione>
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

    def leggi_atto_plus(self, id_documento: str) -> Tuple[List[Tuple[bytes, str]], Dict[str, Any]]:
        """
        Recupera l'atto tramite LeggiAttoPlus ed estrae TUTTI gli allegati e i metadati generali.
        Ritorna: (lista_allegati, sidecar_attributes) dove lista_allegati è List[Tuple[bytes, file_name]]
        """
        filtro_xml = f"""<FiltroAttoIn>
  <IdDocumento>{id_documento}</IdDocumento>
  <DownloadAllegati>S</DownloadAllegati>
  <DownloadAllegatiSenzaFirma>N</DownloadAllegatiSenzaFirma>
</FiltroAttoIn>"""

        soap = self._soap_envelope_leggi_atto_plus(filtro_xml)
        log.debug("Dati SOAP per LeggiAttoPlus:\n%s", soap)
        resp_xml = self._post_soap(soap)

        tree = ET.fromstring(resp_xml)
        
        result_el = None
        for el in tree.iter():
            if el.tag.endswith("LeggiAttoPlusResult"):
                result_el = el
                break

        if result_el is None or not (result_el.text or "").strip():
            raise RuntimeError("Tag LeggiAttoPlusResult vuoto o non trovato nella risposta SOAP.")

        raw_inner = result_el.text.strip()
        decoded_inner = html.unescape(raw_inner)
        inner_tree = ET.fromstring(decoded_inner)

        errore = inner_tree.findtext(".//Errore")
        if errore and errore.strip():
            raise RuntimeError(f"WSAttiSoap Errore Applicativo: {errore.strip()}")

        oggetto = inner_tree.findtext(".//Oggetto") or ""
        numero_atto = ""
        data_atto = ""

        determina_el = inner_tree.find(".//Determina")
        if determina_el is not None:
            numero_atto = determina_el.findtext(".//Numero") or ""
            data_atto = determina_el.findtext(".//Data") or ""
        else:
            delibera_el = inner_tree.find(".//Delibera")
            if delibera_el is not None:
                numero_atto = delibera_el.findtext(".//Numero") or ""
                data_atto = delibera_el.findtext(".//Data") or ""

        allegati_el = inner_tree.findall(".//Allegati/Allegato")
        if not allegati_el:
            raise RuntimeError(f"Nessun allegato binario associato all'atto UID {id_documento}")

        lista_allegati = []
        for idx, allg in enumerate(allegati_el):
            file_name = allg.findtext(".//NomeAllegato")
            file_name = file_name.strip() if file_name else f"allegato_{idx}.bin"

            b64_image = allg.findtext(".//Image")
            if not b64_image or not b64_image.strip():
                log.warning("Payload <Image> vuoto per l'allegato '%s' (UID %s). Saltato.", file_name, id_documento)
                continue

            try:
                file_bytes = base64.b64decode(b64_image.strip())
                lista_allegati.append((file_bytes, file_name))
            except Exception as e:
                log.error("Errore decodifica Base64 per allegato '%s': %s", file_name, str(e))

        if not lista_allegati:
            raise RuntimeError(f"Nessun allegato valido estratto per l'atto UID {id_documento}")

        attributes = {
            "oggetto": oggetto,
            "numero_atto": numero_atto,
            "data_atto": data_atto
        }

        return lista_allegati, attributes

def build_client_from_env() -> WSAttiSoapClient:
    creds = WSAttiCredentials(
        username=settings.repwss_username,
        password=settings.repwss_password,
    )
    return WSAttiSoapClient(
        endpoint_url=settings.docws_atti_endpoint,
        credentials=creds,
        timeout=settings.http_timeout_seconds,
        verify_tls=settings.verify_tls,
    )
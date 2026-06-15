from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, datetime
from enum import Enum
from typing import Optional, Union, List
import xml.etree.ElementTree as ET

DateLike = Union[str, date, datetime]

def _format_date(value: Optional[DateLike]) -> Optional[str]:
    if value is None:
        return None
    if isinstance(value, (datetime, date)):
        return value.strftime("%d/%m/%Y")
    if isinstance(value, str):
        s = value.strip()
        return s if s else None
    raise TypeError(f"Unsupported date type: {type(value)}")

def _add_text_child(parent: ET.Element, tag: str, value: Optional[str]) -> None:
    if value is None:
        return
    v = value.strip()
    if not v:
        return
    child = ET.SubElement(parent, tag)
    child.text = v

def _indent(elem: ET.Element, level: int = 0) -> None:
    i = "\n" + level * "  "
    if len(elem):
        if not (elem.text and elem.text.strip()):
            elem.text = i + "  "
        for child in elem:
            _indent(child, level + 1)
        if not (elem.tail and elem.tail.strip()):
            elem.tail = i
    else:
        if not (elem.tail and elem.tail.strip()):
            elem.tail = i

@dataclass
class DeterminaFilter:
    determina_anno: Optional[str] = None
    determina_numero: Optional[str] = None
    determina_numero_a: Optional[str] = None
    determina_data: Optional[DateLike] = None
    determina_data_a: Optional[DateLike] = None
    determina_anno_gen: Optional[str] = None
    determina_numero_gen: Optional[str] = None
    determina_numero_gen_a: Optional[str] = None
    determina_data_gen: Optional[DateLike] = None
    determina_data_gen_a: Optional[DateLike] = None
    trattamento: Optional[str] = None
    dirigente: Optional[str] = None
    ufficio: Optional[str] = None
    pubblicazione_data: Optional[DateLike] = None
    pubblicazione_data_a: Optional[DateLike] = None
    esecutivita_data: Optional[DateLike] = None
    esecutivita_data_a: Optional[DateLike] = None
    adozione_data: Optional[DateLike] = None
    adozione_data_a: Optional[DateLike] = None

    def has_active_filters(self) -> bool:
        return any(v is not None and str(v).strip() != "" for v in vars(self).values())

    def to_xml_element(self) -> ET.Element:
        det_el = ET.Element("Determina")
        _add_text_child(det_el, "DeterminaAnno", self.determina_anno)
        _add_text_child(det_el, "DeterminaNumero", self.determina_numero)
        _add_text_child(det_el, "DeterminaNumeroA", self.determina_numero_a)
        _add_text_child(det_el, "DeterminaData", _format_date(self.determina_data))
        _add_text_child(det_el, "DeterminaDataA", _format_date(self.determina_data_a))
        _add_text_child(det_el, "DeterminaAnnoGen", self.determina_anno_gen)
        _add_text_child(det_el, "DeterminaNumeroGen", self.determina_numero_gen)
        _add_text_child(det_el, "DeterminaNumeroGenA", self.determina_numero_gen_a)
        _add_text_child(det_el, "DeterminaDataGen", _format_date(self.determina_data_gen))
        _add_text_child(det_el, "DeterminaDataGenA", _format_date(self.determina_data_gen_a))
        _add_text_child(det_el, "Trattamento", self.trattamento)
        _add_text_child(det_el, "Dirigente", self.dirigente)
        _add_text_child(det_el, "Ufficio", self.ufficio)
        _add_text_child(det_el, "PubblicazioneData", _format_date(self.pubblicazione_data))
        _add_text_child(det_el, "PubblicazioneDataA", _format_date(self.pubblicazione_data_a))
        _add_text_child(det_el, "EsecutivitaData", _format_date(self.esecutivita_data))
        _add_text_child(det_el, "EsecutivitaDataA", _format_date(self.esecutivita_data_a))
        _add_text_child(det_el, "AdozioneData", _format_date(self.adozione_data))
        _add_text_child(det_el, "AdozioneDataA", _format_date(self.adozione_data_a))
        return det_el

@dataclass
class DeliberaFilter:
    delibera_anno: Optional[str] = None
    delibera_numero: Optional[str] = None
    delibera_numero_a: Optional[str] = None
    delibera_data: Optional[DateLike] = None
    delibera_data_a: Optional[DateLike] = None
    trattamento: Optional[str] = None
    dirigente: Optional[str] = None
    ufficio: Optional[str] = None
    pubblicazione_data: Optional[DateLike] = None
    pubblicazione_data_a: Optional[DateLike] = None
    esecutivita_data: Optional[DateLike] = None
    esecutivita_data_a: Optional[DateLike] = None
    adozione_data: Optional[DateLike] = None
    adozione_data_a: Optional[DateLike] = None

    def has_active_filters(self) -> bool:
        return any(v is not None and str(v).strip() != "" for v in vars(self).values())

    def to_xml_element(self) -> ET.Element:
        delib_el = ET.Element("Delibera")
        _add_text_child(delib_el, "DeliberaAnno", self.delibera_anno)
        _add_text_child(delib_el, "DeliberaNumero", self.delibera_numero)
        _add_text_child(delib_el, "DeliberaNumeroA", self.delibera_numero_a)
        _add_text_child(delib_el, "DeliberaData", _format_date(self.delibera_data))
        _add_text_child(delib_el, "DeliberaDataA", _format_date(self.delibera_data_a))
        _add_text_child(delib_el, "Trattamento", self.trattamento)
        _add_text_child(delib_el, "Dirigente", self.dirigente)
        _add_text_child(delib_el, "Ufficio", self.ufficio)
        _add_text_child(delib_el, "PubblicazioneData", _format_date(self.pubblicazione_data))
        _add_text_child(delib_el, "PubblicazioneDataA", _format_date(self.pubblicazione_data_a))
        _add_text_child(delib_el, "EsecutivitaData", _format_date(self.esecutivita_data))
        _add_text_child(delib_el, "EsecutivitaDataA", _format_date(self.esecutivita_data_a))
        _add_text_child(delib_el, "AdozioneData", _format_date(self.adozione_data))
        _add_text_child(delib_el, "AdozioneDataA", _format_date(self.adozione_data_a))
        return delib_el

@dataclass
class DecretoFilter:
    decreto_anno: Optional[str] = None
    decreto_numero: Optional[str] = None
    decreto_numero_a: Optional[str] = None
    decreto_data: Optional[DateLike] = None
    decreto_data_a: Optional[DateLike] = None
    trattamento: Optional[str] = None
    dirigente: Optional[str] = None
    ufficio: Optional[str] = None
    pubblicazione_data: Optional[DateLike] = None
    pubblicazione_data_a: Optional[DateLike] = None
    esecutivita_data: Optional[DateLike] = None
    esecutivita_data_a: Optional[DateLike] = None
    adozione_data: Optional[DateLike] = None
    adozione_data_a: Optional[DateLike] = None

    def has_active_filters(self) -> bool:
        return any(v is not None and str(v).strip() != "" for v in vars(self).values())

    def to_xml_element(self) -> ET.Element:
        decr_el = ET.Element("Decreto")
        _add_text_child(decr_el, "DecretoAnno", self.decreto_anno)
        _add_text_child(decr_el, "DecretoNumero", self.decreto_numero)
        _add_text_child(decr_el, "DecretoNumeroA", self.decreto_numero_a)
        _add_text_child(decr_el, "DecretoData", _format_date(self.decreto_data))
        _add_text_child(decr_el, "DecretoDataA", _format_date(self.decreto_data_a))
        _add_text_child(decr_el, "Trattamento", self.trattamento)
        _add_text_child(decr_el, "Dirigente", self.dirigente)
        _add_text_child(decr_el, "Ufficio", self.ufficio)
        _add_text_child(decr_el, "PubblicazioneData", _format_date(self.pubblicazione_data))
        _add_text_child(decr_el, "PubblicazioneDataA", _format_date(self.pubblicazione_data_a))
        _add_text_child(decr_el, "EsecutivitaData", _format_date(self.esecutivita_data))
        _add_text_child(decr_el, "EsecutivitaDataA", _format_date(self.esecutivita_data_a))
        _add_text_child(decr_el, "AdozioneData", _format_date(self.adozione_data))
        _add_text_child(decr_el, "AdozioneDataA", _format_date(self.adozione_data_a))
        return decr_el

@dataclass
class MetadataItem:
    nome_tabella: str
    nome_campo: str
    valore_campo: str

    def validate(self) -> None:
        if not self.nome_tabella or not self.nome_tabella.strip(): raise ValueError("NomeTabella mancante")
        if not self.nome_campo or not self.nome_campo.strip(): raise ValueError("NomeCampo mancante")
        if self.valore_campo is None or not str(self.valore_campo).strip(): raise ValueError("ValoreCampo mancante")

    def to_xml_element(self) -> ET.Element:
        self.validate()
        item_el = ET.Element("item")
        _add_text_child(item_el, "NomeTabella", self.nome_tabella)
        _add_text_child(item_el, "NomeCampo", self.nome_campo)
        _add_text_child(item_el, "ValoreCampo", str(self.valore_campo))
        return item_el

@dataclass
class DatiUtenteFilter:
    items: List[MetadataItem] = field(default_factory=list)

    def has_active_filters(self) -> bool:
        return len(self.items) > 0

    def to_xml_element(self) -> ET.Element:
        du_el = ET.Element("DatiUtente")
        for it in self.items: du_el.append(it.to_xml_element())
        return du_el

@dataclass
class RicercaFiltri:
    determina: Optional[DeterminaFilter] = None
    delibera: Optional[DeliberaFilter] = None
    decreto: Optional[DecretoFilter] = None
    dati_utente: Optional[DatiUtenteFilter] = None
    utente: Optional[str] = None
    ruolo: Optional[str] = None

    def validate(self) -> None:
        if not any([self.determina, self.delibera, self.decreto, self.dati_utente]):
            raise ValueError("Configurare almeno una sezione di ricerca (determina, delibera, decreto, dati_utente)")
        
        has_active = False
        for filter_obj in [self.determina, self.delibera, self.decreto, self.dati_utente]:
            if filter_obj and filter_obj.has_active_filters():
                has_active = True
                break
                
        if not has_active:
            raise ValueError("Ricerca bloccata: nessun parametro di ricerca valorizzato.")

    def to_xml(self, pretty: bool = True, xml_declaration: bool = False, encoding: str = "utf-8") -> str:
        self.validate()
        root = ET.Element("RicercaFiltri")
        
        # Gestiamo l'aggiunta in base a cosa è stato istanziato
        if self.determina: root.append(self.determina.to_xml_element())
        if self.delibera: root.append(self.delibera.to_xml_element())
        if self.decreto: root.append(self.decreto.to_xml_element())
        if self.dati_utente: root.append(self.dati_utente.to_xml_element())
            
        _add_text_child(root, "Utente", self.utente)
        _add_text_child(root, "Ruolo", self.ruolo)
        
        if pretty: _indent(root)
        return ET.tostring(root, encoding=encoding, xml_declaration=xml_declaration).decode(encoding)
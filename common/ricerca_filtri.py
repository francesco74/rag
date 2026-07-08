from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, datetime, timedelta
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

def _parse_date_like(value: DateLike) -> date:
    """Converte una DateLike (str 'dd/mm/YYYY', date o datetime) in un oggetto date,
    usato per iterare giorno per giorno su un range (vedi RicercaSemplice)."""
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, date):
        return value
    if isinstance(value, str):
        s = value.strip()
        if not s:
            raise ValueError("Data vuota non convertibile")
        return datetime.strptime(s, "%d/%m/%Y").date()
    raise TypeError(f"Formato data non supportato: {type(value)}")

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
class DeliberaFilter:
    delibera_anno: Optional[str] = None
    delibera_numero: Optional[str] = None
    delibera_data: Optional[DateLike] = None
    organo: Optional[str] = None  # Es: 1=consiglio, 2=giunta, 7=commissario
    trattamento: Optional[str] = None
    relatore: Optional[str] = None
    pubblicazione_data: Optional[DateLike] = None
    esecutivita_data: Optional[DateLike] = None
    immediata_esecutivita: Optional[str] = None  # S/N
    capigruppo: Optional[str] = None
    prefetto: Optional[str] = None
    coreco: Optional[str] = None

    def has_active_filters(self) -> bool:
        return any(v is not None and str(v).strip() != "" for v in vars(self).values())

    def to_xml_element(self) -> ET.Element:
        delib_el = ET.Element("Delibera")
        _add_text_child(delib_el, "DeliberaAnno", self.delibera_anno)
        _add_text_child(delib_el, "DeliberaNumero", self.delibera_numero)
        _add_text_child(delib_el, "DeliberaData", _format_date(self.delibera_data))
        _add_text_child(delib_el, "Organo", self.organo)
        _add_text_child(delib_el, "Trattamento", self.trattamento)
        _add_text_child(delib_el, "Relatore", self.relatore)
        _add_text_child(delib_el, "PubblicazioneData", _format_date(self.pubblicazione_data))
        _add_text_child(delib_el, "EsecutivitaData", _format_date(self.esecutivita_data))
        _add_text_child(delib_el, "ImmediataEsecutivita", self.immediata_esecutivita)
        _add_text_child(delib_el, "Capigruppo", self.capigruppo)
        _add_text_child(delib_el, "Prefetto", self.prefetto)
        _add_text_child(delib_el, "Coreco", self.coreco)
        return delib_el
    

@dataclass
class DeterminaFilter:
    oggetto: Optional[str] = None                    
    tipo_ricerca_oggetto: Optional[str] = "T"         # <--  (Default AND)
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
class DecretoFilter:
    tipo_decreto: Optional[str] = "deliberativo"
    decreto_numero: Optional[str] = None
    decreto_data: Optional[DateLike] = None
    oggetto: Optional[str] = None
    tipo_ricerca_oggetto: Optional[str] = "T"

    def has_active_filters(self) -> bool:
        return any(v is not None and str(v).strip() != "" for v in vars(self).values())

    def to_xml_element(self) -> ET.Element:
        doc_el = ET.Element("Documento")
        
        # CORRETTO da log reale: una ricerca con tipo_decreto="deliberativo"
        # (quindi <Tipo>DEC</Tipo>) ha restituito un decreto PRESIDENZIALE
        # (il cui unico registro definitivo è "Registro Verbale (DEC_VBMP)").
        # Questo dimostra che <Tipo> in <Documento> NON distingue affatto
        # deliberativo da presidenziale (probabilmente "DEP" non è nemmeno un
        # codice valido lato server: era un'ipotesi mai confermata). Entrambi
        # i sottotipi condividono lo stesso TipoDocumento "DEC" nella
        # risposta. La distinzione deliberativo/presidenziale va quindi fatta
        # SOLO lato client, controllando quale registro verbale è presente
        # nella risposta (DEC_VBDD vs DEC_VBMP) — vedi verify_pipeline in
        # verifica.py, che applica questo filtro dopo la ricerca.
        MAPPING_DECRETI = {
            "deliberativo": "DEC",
            "presidenziale": "DEC",
        }
        user_tipo = self.tipo_decreto or "deliberativo"
        codice_soap = MAPPING_DECRETI.get(user_tipo.lower(), "DEC")
        
        _add_text_child(doc_el, "Numero", self.decreto_numero)
        _add_text_child(doc_el, "Data", _format_date(self.decreto_data))
        _add_text_child(doc_el, "Tipo", codice_soap)
        _add_text_child(doc_el, "Oggetto", self.oggetto)
        if self.oggetto:
            _add_text_child(doc_el, "TipoRicercaOggetto", self.tipo_ricerca_oggetto or "T")
            
        return doc_el
            
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
    # Ricerca per SOLO oggetto, TRASVERSALE a tutti i tipi di atto (nessun <Tipo>
    # emesso in <Documento>). CONFERMATO da log reale: una ricerca con solo
    # <Documento><Oggetto>...</Oggetto><TipoRicercaOggetto>T</TipoRicercaOggetto></Documento>
    # (senza <Tipo>) ha restituito indifferentemente una delibera. Da usare
    # SOLO in alternativa a determina/delibera/decreto, non insieme.
    oggetto_libero: Optional[str] = None
    tipo_ricerca_oggetto_libero: Optional[str] = "T"

    def validate(self) -> None:
        if not any([self.determina, self.delibera, self.decreto, self.dati_utente, self.oggetto_libero]):
            raise ValueError("Configurare almeno una sezione di ricerca (determina, delibera, decreto, dati_utente, oggetto_libero)")
        
        has_active = False
        for filter_obj in [self.determina, self.delibera, self.decreto, self.dati_utente]:
            if filter_obj and filter_obj.has_active_filters():
                has_active = True
                break

        if self.oggetto_libero and self.oggetto_libero.strip():
            has_active = True

        if not has_active:
            raise ValueError("Ricerca bloccata: nessun parametro di ricerca valorizzato.")

    def to_xml(self, pretty: bool = True, xml_declaration: bool = False, encoding: str = "utf-8") -> str:
        self.validate()
        root = ET.Element("RicercaFiltri")
        
        # SEQUENZA STRUTTURALE WSDL OBBLIGATORIA: Documento -> Delibera -> Determina
        
        # 1. Sezione <Documento> (Generata nativamente da DecretoFilter, oppure
        #    generica per ricerca "solo oggetto" trasversale a tutti i tipi)
        if self.decreto:
            root.append(self.decreto.to_xml_element())
        elif self.determina and hasattr(self.determina, "oggetto") and self.determina.oggetto:
            doc_el = ET.Element("Documento")
            _add_text_child(doc_el, "Tipo", "DET")
            _add_text_child(doc_el, "Oggetto", self.determina.oggetto)
            root.append(doc_el)
        elif self.oggetto_libero:
            # CONFERMATO da log reale: nessun <Tipo> -> ricerca su tutti i tipi di atto
            doc_el = ET.Element("Documento")
            _add_text_child(doc_el, "Oggetto", self.oggetto_libero)
            _add_text_child(doc_el, "TipoRicercaOggetto", self.tipo_ricerca_oggetto_libero or "T")
            root.append(doc_el)
            
        # 2. Sezione <Delibera>
        if self.delibera:
            root.append(self.delibera.to_xml_element())
            
        # 3. Sezione <Determina>
        if self.determina:
            root.append(self.determina.to_xml_element())
            
        # 4. Sezione <DatiUtente>
        if self.dati_utente:
            root.append(self.dati_utente.to_xml_element())
            
        _add_text_child(root, "Utente", self.utente)
        _add_text_child(root, "Ruolo", self.ruolo)
        
        if pretty: _indent(root)
        return ET.tostring(root, encoding=encoding, xml_declaration=xml_declaration).decode(encoding)


# ---------------------------------------------------------------------------
# Mappa dei "registri definitivi" (quelli che valgono come numero/anno ufficiali
# dell'atto, escludendo il Registro Proposta). Confermati da log reali:
#   - determina                          -> Registro Verbale (DET_VB)
#   - decreto,    tipo_decreto=deliberativo -> Registro Verbale (DEC_VBDD)
#   - decreto,    tipo_decreto=presidenziale -> Registro Verbale (DEC_VBMP)
#   - delibera,   organo=consiglio (1)    -> Registro Verbale (DLC_VB)
#
# NON confermati (ipotesi da verificare con un log reale prima di fare affidamento):
#   - delibera di Giunta (organo=2)      -> ipotizzato "DLG_VB"
#   - delibera Commissario (organo=7)    -> ipotizzato "DLM_VB" (o simile)
# ---------------------------------------------------------------------------
REGISTRO_DEFINITIVO_CODICI = {
    "determina": "DET_VB",
    "delibera": {
        "1": "DLC_VB",          # consiglio - CONFERMATO
        "2": "DLG_VB",          # giunta - DA VERIFICARE (non confermato da log)
        "7": "DLM_VB",          # commissario - DA VERIFICARE (non confermato da log)
        None: "DLC_VB",         # default: consiglio, se organo non specificato
    },
    "decreto": {
        "deliberativo": "DEC_VBDD",
        "presidenziale": "DEC_VBMP",
    },
}


def tutti_i_codici_registro_definitivo() -> list[str]:
    """Elenco piatto di TUTTI i codici registro 'definitivo' noti (determina +
    ogni sotto-tipo di delibera/decreto), usato dal parser dei risultati per
    l'auto-detect quando non si conosce a priori il tipo di atto cercato
    (es. ricerca trasversale per solo oggetto, tipo_atto='qualsiasi')."""
    codici = [REGISTRO_DEFINITIVO_CODICI["determina"]]
    codici += list(dict.fromkeys(REGISTRO_DEFINITIVO_CODICI["delibera"].values()))
    codici += list(REGISTRO_DEFINITIVO_CODICI["decreto"].values())
    return codici


@dataclass
class RicercaSemplice:
    """
    Facciata semplificata per le ricerche: solo oggetto, range temporale
    (anche di un solo giorno) e numero di atto. Traduce questi parametri
    nel RicercaFiltri "ricco" già esistente, usando per ciascun tipo di atto
    i campi che rappresentano il registro DEFINITIVO (quello che interessa
    all'utente), non il registro Proposta.

    tipo_atto="qualsiasi": ricerca SOLO OGGETTO, trasversale a tutti i tipi
    di atto (determina/delibera/decreto insieme) — CONFERMATO da log reale:
    <Documento><Oggetto>...</Oggetto><TipoRicercaOggetto>T</TipoRicercaOggetto></Documento>
    senza <Tipo> restituisce indifferentemente atti di qualunque tipo.
    In questa modalità numero_atto/anno_atto NON sono supportati (non è
    confermato come il WSDL li combinerebbe con un <Documento> privo di
    <Tipo>) e vengono ignorati con un warning.

    tipo_atto="decreto_deliberativo" / "decreto_presidenziale": NON esiste
    più un parametro separato tipo_decreto. CONFERMATO da log reale che
    <Tipo>DEC</Tipo> nella richiesta SOAP non distingue affatto i due
    sotto-tipi (una ricerca con tipo_decreto="deliberativo" ha restituito un
    decreto il cui unico registro era "Registro Verbale (DEC_VBMP)",
    cioè presidenziale) — quindi il sotto-tipo va trattato come un tipo_atto
    a sé stante fin dall'inizio, e la sua verifica avviene SEMPRE lato
    client dopo la ricerca, confrontando il registro definitivo
    effettivamente trovato con quello atteso (vedi registro_definitivo_atteso
    e il filtro applicato in verify_pipeline, in verifica.py).
    """
    tipo_atto: str                        # "determina" | "delibera" | "decreto_deliberativo" | "decreto_presidenziale" | "qualsiasi"
    oggetto: Optional[str] = None
    numero_atto: Optional[str] = None
    anno_atto: Optional[str] = None
    data_da: Optional[DateLike] = None
    data_a: Optional[DateLike] = None     # se assente e data_da presente -> un solo giorno
    organo: Optional[str] = None          # 1=consiglio, 2=giunta, 7=commissario (solo se tipo_atto == "delibera")

    TIPI_VALIDI = {"determina", "delibera", "decreto_deliberativo", "decreto_presidenziale", "qualsiasi"}

    def __post_init__(self):
        if self.tipo_atto not in self.TIPI_VALIDI:
            raise ValueError(f"tipo_atto deve essere uno tra: {sorted(self.TIPI_VALIDI)}")
        if self.data_da and not self.data_a:
            self.data_a = self.data_da  # un solo giorno
        if self.tipo_atto == "qualsiasi":
            if not self.oggetto:
                raise ValueError("tipo_atto='qualsiasi' richiede almeno 'oggetto' (ricerca solo per oggetto, trasversale a tutti i tipi)")
            if self.numero_atto or self.anno_atto or self.data_da:
                import logging as _logging
                _logging.getLogger(__name__).warning(
                    "RicercaSemplice(tipo_atto='qualsiasi'): numero_atto/anno_atto/data_da/data_a "
                    "vengono ignorati perché non confermato come combinarli con una ricerca "
                    "trasversale senza <Tipo>. Specificare un tipo_atto esplicito per usarli."
                )

    def registro_definitivo_atteso(self) -> Optional[str]:
        """Restituisce il codice del registro (es. 'DEC_VBDD') che verrà
        considerato 'definitivo' per questa ricerca, usato poi dal parser
        dei risultati per estrarre numero/anno corretti. Per tipo_atto=
        'qualsiasi' restituisce None: il parser individua il registro
        corretto documento per documento, scandagliando tutti i codici noti."""
        if self.tipo_atto == "qualsiasi":
            return None
        if self.tipo_atto == "determina":
            return REGISTRO_DEFINITIVO_CODICI["determina"]
        if self.tipo_atto == "decreto_deliberativo":
            return REGISTRO_DEFINITIVO_CODICI["decreto"]["deliberativo"]
        if self.tipo_atto == "decreto_presidenziale":
            return REGISTRO_DEFINITIVO_CODICI["decreto"]["presidenziale"]
        if self.tipo_atto == "delibera":
            return REGISTRO_DEFINITIVO_CODICI["delibera"].get(
                self.organo, REGISTRO_DEFINITIVO_CODICI["delibera"][None]
            )
        raise ValueError(f"tipo_atto sconosciuto: {self.tipo_atto}")

    def to_filtri_list(self, utente: str = "utente@wsprotocollo", ruolo: str = "CED") -> List[RicercaFiltri]:
        """
        Restituisce una LISTA di RicercaFiltri da eseguire (con run_search)
        e i cui risultati vanno poi uniti.

        - determina: un solo RicercaFiltri, perché DeterminaFilter ha già un
          range nativo confermato (determina_data_gen / determina_data_gen_a).
        - delibera / decreto_*: DeliberaFilter e DecretoFilter hanno solo un
          campo data secco (nessun "_a" confermato dal WSDL), quindi un range
          si ottiene lanciando una ricerca per OGNI giorno del range e unendo
          i risultati (vedi verify_pipeline in verifica.py).
        """
        if self.tipo_atto == "determina":
            return [self._filtri_determina(utente, ruolo)]

        if self.tipo_atto == "qualsiasi":
            return [RicercaFiltri(oggetto_libero=self.oggetto, utente=utente, ruolo=ruolo)]

        # delibera / decreto_deliberativo / decreto_presidenziale: un giorno
        # solo se non c'è range, altrimenti loop giornaliero
        if not self.data_da:
            return [self._filtri_giorno(utente, ruolo, giorno=None)]

        giorno_da = _parse_date_like(self.data_da)
        giorno_a = _parse_date_like(self.data_a) if self.data_a else giorno_da
        if giorno_a < giorno_da:
            raise ValueError("data_a non può essere precedente a data_da")

        filtri_list = []
        giorno = giorno_da
        while giorno <= giorno_a:
            filtri_list.append(self._filtri_giorno(utente, ruolo, giorno=giorno))
            giorno += timedelta(days=1)
        return filtri_list

    def _filtri_determina(self, utente: str, ruolo: str) -> RicercaFiltri:
        # I campi "_gen" sono quelli che, nel resto della codebase (vedi
        # retrigger_extraction), rappresentano già il registro verbale/definitivo.
        determina = DeterminaFilter(
            oggetto=self.oggetto,
            determina_numero_gen=self.numero_atto,
            determina_anno_gen=self.anno_atto,
            determina_data_gen=self.data_da,
            determina_data_gen_a=self.data_a,
        )
        return RicercaFiltri(determina=determina, utente=utente, ruolo=ruolo)

    def _filtri_giorno(self, utente: str, ruolo: str, giorno: Optional[date]) -> RicercaFiltri:
        kwargs = {"utente": utente, "ruolo": ruolo}

        if self.tipo_atto == "delibera":
            kwargs["delibera"] = DeliberaFilter(
                delibera_numero=self.numero_atto,
                delibera_anno=self.anno_atto,
                delibera_data=giorno,
                organo=self.organo,
            )
            # Nota: DeliberaFilter non ha un campo "oggetto" nativo nel WSDL.
            # Se serve filtrare le delibere anche per oggetto, va aggiunto un
            # campo dedicato (tag SOAP da confermare).

        elif self.tipo_atto in ("decreto_deliberativo", "decreto_presidenziale"):
            # NOTA: tipo_decreto qui passato a DecretoFilter è ininfluente sul
            # risultato della ricerca (<Tipo>DEC</Tipo> è identico per entrambi
            # i sotto-tipi, vedi docstring della classe) — lo passiamo solo
            # per chiarezza/leggibilità dell'XML generato, non perché filtri
            # davvero qualcosa lato server. Il filtro vero è client-side.
            tipo_decreto_interno = "deliberativo" if self.tipo_atto == "decreto_deliberativo" else "presidenziale"
            kwargs["decreto"] = DecretoFilter(
                tipo_decreto=tipo_decreto_interno,
                decreto_numero=self.numero_atto,
                decreto_data=giorno,
                oggetto=self.oggetto,
            )

        return RicercaFiltri(**kwargs)
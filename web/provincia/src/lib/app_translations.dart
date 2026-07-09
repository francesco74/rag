import 'package:flutter/material.dart';

// 1. Define the supported languages
enum AppLang { it, en }

// 2. Create the Global Notifier (defaults to Italian)
final ValueNotifier<AppLang> langNotifier = ValueNotifier(AppLang.it);

// 3. Create the Translation Dictionary
class AppTranslations {
  static const Map<String, Map<AppLang, String>> _strings = {
    'welcome_msg': {
      AppLang.it:
          "<p>Ciao! Chiedimi qualunque cosa sui documenti che conosco.</p>",
      AppLang.en: "<p>Hi! Ask me anything about the documents I know.</p>",
    },
    'ask_hint': {
      AppLang.it: "Sottoponi la domanda...",
      AppLang.en: "Ask a question...",
    },
    'clear_chat': {AppLang.it: "Svuota Chat", AppLang.en: "Clear Chat"},
    'chat_cleared': {
      AppLang.it: "Cronologia chat cancellata.",
      AppLang.en: "Chat history cleared.",
    },
    'sources': {
      AppLang.it: "Fonti analizzate:",
      AppLang.en: "Analyzed sources:",
    },
    'unknown_file': {
      AppLang.it: "File sconosciuto",
      AppLang.en: "Unknown file",
    },
    'welcome_html_path': {
      AppLang.it: "assets/html/welcome_it.html",
      AppLang.en: "assets/html/welcome_en.html",
    },
    'timeout': {
      AppLang.it: "La richiesta ha impiegato troppo tempo. Riprova.",
      AppLang.en: "The request took too long. Please try again.",
    },
    'connection_lost': {
      AppLang.it: "Connessione persa. Riprova.",
      AppLang.en: "Connection lost. Please try again.",
    },
    'feedback_error': {
      AppLang.it: "Errore nell'invio del feedback. Riprova.",
      AppLang.en: "Error sending feedback. Please try again.",
    },
    'document_error': {
      AppLang.it: "Errore nell'elaborazione del documento.",
      AppLang.en: "Error processing the document.",
    },
    'good_response': {
      AppLang.it: "Buona risposta",
      AppLang.en: "Good response",
    },
    'bad_response': {AppLang.it: "Risposta errata", AppLang.en: "Bad response"},
    'unknown_topic': {
      AppLang.it: "argomento sconosciuto",
      AppLang.en: "unknown topic",
    },
    'error': {AppLang.it: "Errore", AppLang.en: "Error"},
    'unknown_error': {
      AppLang.it: "Errore sconosciuto",
      AppLang.en: "Unknown error",
    },
    'feedback_received': {
      AppLang.it: "Grazie per il tuo feedback!",
      AppLang.en: "Thanks for your feedback!",
    },
    'toggle_theme': {AppLang.it: "Cambia tema", AppLang.en: "Toggle Theme"},
    'delete_chat': {AppLang.it: "Elimina Chat", AppLang.en: "Delete Chat"},
    'welcome_load_error': {
      AppLang.it: "Errore nel caricamento del messaggio di benvenuto.",
      AppLang.en: "Error loading welcome message.",
    },
    'like_msg': {
      AppLang.it:
          "Felice di esserti stato d'aiuto! Vuoi aggiungere un commento?",
      AppLang.en: "Glad it helped! Would you like to add a comment?",
    },
    'dislike_msg': {
      AppLang.it:
          "Mi dispiace che la risposta non sia stata utile. Vuoi aggiungere un commento per aiutarci a migliorare?",
      AppLang.en:
          "Sorry to hear the response wasn't helpful. Would you like to add a comment to help us improve?",
    },
    'comment_hint': {
      AppLang.it: "Aggiungi un commento opzionale...",
      AppLang.en: "Add an optional comment...",
    },
    'filter_subtopics': {
      AppLang.it: "Filtra Sottocategorie",
      AppLang.en: "Filter Subtopics",
    },
    'subtopics_title': {AppLang.it: "Sottocategorie", AppLang.en: "Subtopics"},
    'maintenance_mode': {
      AppLang.it: "Sistema in manutenzione",
      AppLang.en: "Maintenance Mode",
    },
    'maintenance_msg': {
      AppLang.it:
          "Stiamo effettuando degli aggiornamenti tecnici. Il servizio tornerà disponibile a breve.",
      AppLang.en:
          "We are performing technical updates. The service will be back shortly.",
    },
    'retry_connection': {
      AppLang.it: "Riprova a collegarti",
      AppLang.en: "Retry Connection",
    },
    'skip': {AppLang.it: "Salta", AppLang.en: "Skip"},
    'submit': {AppLang.it: "Invia", AppLang.en: "Submit"},
    'change_language': {
      AppLang.it: "Cambio Lingua",
      AppLang.en: "Change Language",
    },
    'change_language_desc': {
      AppLang.it:
          "Passa dall'italiano all'inglese. L'IA tradurrà automaticamente i documenti per te.",
      AppLang.en:
          "Switch between Italian and English. The AI will automatically translate documents for you.",
    },
    'change_theme': {
      AppLang.it: "Modalità Lettura",
      AppLang.en: "Reading Mode",
    },
    'change_theme_desc': {
      AppLang.it:
          "Cambia i colori dello schermo (Chiaro, Scuro o Alto Contrasto) per migliorare la visibilità.",
      AppLang.en:
          "Change the screen colors (Light, Dark, or High Contrast) to improve visibility.",
    },
    'delete_chat_desc': {
      AppLang.it:
          "Elimina la chat corrente e inizia una nuova conversazione.",
      AppLang.en:
          "Delete the current chat and start a new conversation.",
    },
    'filter_subtopics_desc': {
      AppLang.it:
          "Filtro per effettuare le ricerche solamente sulle serie che ti interessano.",
      AppLang.en: "Filter to search only within the series that interest you.",
    },
    'ai_processing': {
      AppLang.it: "Sto elaborando la tua domanda...",
      AppLang.en: "Processing your question...",
    },
    'taking_longer': {
      AppLang.it:
          "La ricerca è complessa, ci sta volendo più del previsto...attendi ancora un attimo...",
      AppLang.en:
          "The search is complex, it's taking longer than expected...please wait a bit more...",
    },
    'send_question': {
      AppLang.it: "Invia domanda",
      AppLang.en: "Send question",
    },
    'send_question_desc': {
      AppLang.it: "Invia la tua domanda al chatbot",
      AppLang.en: "Send your question to the chatbot",
    },

    'date_filter_title': {
      AppLang.it: "Filtro per data",
      AppLang.en: "Date Filter",
    },
    'date_from': {
      AppLang.it: "Data da",
      AppLang.en: "Date from",
    },
    'date_to': {
      AppLang.it: "Data a",
      AppLang.en: "Date to",
    },
    'clear_date_filter': {
      AppLang.it: "Cancella filtro data",
      AppLang.en: "Clear date filter",
    },
    'include_undated_docs': {
      AppLang.it: "Includi documenti senza data",
      AppLang.en: "Include undated documents",
    },
    'subtopic_stat_with_year': {
      AppLang.it: "{desc}: {count} documenti, a partire dal {year}",
      AppLang.en: "{desc}: {count} documents, since {year}",
    },
    'subtopic_stat_no_year': {
      AppLang.it: "{desc}: {count} documenti",
      AppLang.en: "{desc}: {count} documents",
    },

  };

  /// Helper function to grab the correct string based on the current language
  static String get(String key, AppLang lang) {
    return _strings[key]?[lang] ?? key; // Returns the key itself if not found
  }
}
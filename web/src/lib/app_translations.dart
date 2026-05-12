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
    // Bonus: You can also translate the path to your welcome HTML file!
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
    'welcome_load_error': {
      AppLang.it: "Errore nel caricamento del messaggio di benvenuto.",
      AppLang.en: "Error loading welcome message.",
    },
    'like_msg': {
      AppLang.it: "Felice di esserti stato d'aiuto! Vuoi aggiungere un commento?",
      AppLang.en: "Glad it helped! Would you like to add a comment?",
    },
    'dislike_msg': {
      AppLang.it: "Mi dispiace che la risposta non sia stata utile. Vuoi aggiungere un commento per aiutarci a migliorare?",
      AppLang.en: "Sorry to hear the response wasn't helpful. Would you like to add a comment to help us improve?", 
    },
    'comment_hint': {
      AppLang.it: "Aggiungi un commento opzionale...",
      AppLang.en: "Add an optional comment...",
    },
    'filter_subtopics': {
      AppLang.it: "Filtra Sottocategorie",
      AppLang.en: "Filter Subtopics",
    },
    'subtopics_title': {
      AppLang.it: "Sottocategorie",
      AppLang.en: "Subtopics",
    },
    'send_question': {
      AppLang.it: "Invia domanda",
      AppLang.en: "Send question",
    },
    'maintenance_mode': {
      AppLang.it: "Sistema in manutenzione",
      AppLang.en: "Maintenance Mode",
    },
    'maintenance_msg': {
      AppLang.it: "Stiamo effettuando degli aggiornamenti tecnici. Il servizio tornerà disponibile a breve.",
      AppLang.en: "We are performing technical updates. The service will be back shortly.",
    },
    'retry_connection': {
      AppLang.it: "Riprova a collegarti",
      AppLang.en: "Retry Connection",
    },
    'skip': {
      AppLang.it: "Salta",
      AppLang.en: "Skip",
    },    'submit': {
      AppLang.it: "Invia",
      AppLang.en: "Submit",
    },
  };

  /// Helper function to grab the correct string based on the current language
  static String get(String key, AppLang lang) {
    return _strings[key]?[lang] ?? key; // Returns the key itself if not found
  }
}

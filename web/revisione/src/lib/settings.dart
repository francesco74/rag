import 'dart:js_interop';

// Configurazione a runtime scritta da docker-entrypoint.sh in env-config.js.
@JS('ENV_CONFIG')
external EnvConfigJS? get envConfigJS;

@JS()
extension type EnvConfigJS._(JSObject _) implements JSObject {
  // ignore: non_constant_identifier_names
  external String? get REVIEW_API_URL;
  // ignore: non_constant_identifier_names
  external String? get PROJECT_NAME;
  // ignore: non_constant_identifier_names
  external String? get FILES_URL;
}

class AppSettings {
  // URL dell'API (app.py) seguito dal prefisso delle rotte di revisione.
  static const String _defaultApiUrl = 'http://127.0.0.1:5000/review';
  static const String _defaultProjectName = 'Revisione documenti';

  /// Può essere assoluto (https://host/api/review) o relativo al sito da
  /// cui si apre la pagina (/api/review): in quel caso si completa con
  /// l'indirizzo corrente, così la stessa immagine funziona su più domini.
  static String get apiUrl {
    final value = envConfigJS?.REVIEW_API_URL;
    var url = (value == null || value.isEmpty) ? _defaultApiUrl : value;
    url = Uri.base.resolve(url).toString();
    return url.endsWith('/') ? url.substring(0, url.length - 1) : url;
  }

  /// Download dei documenti originali (/files dell'API). Se FILES_URL non è
  /// impostato si ricava da REVIEW_API_URL: .../api/review -> .../api/files.
  static String get filesUrl {
    final value = envConfigJS?.FILES_URL;
    if (value != null && value.isNotEmpty) {
      final url = Uri.base.resolve(value).toString();
      return url.endsWith('/') ? url.substring(0, url.length - 1) : url;
    }
    final api = apiUrl;
    // .../api/review -> .../api/files
    return Uri.parse('$api/').resolve('../files').toString();
  }

  /// Verifica di funzionamento: il /health dell'API, lo stesso della chat.
  /// Si ricava da REVIEW_API_URL: .../api/review -> .../api/health.
  static String get healthUrl =>
      Uri.parse('$apiUrl/').resolve('../health').toString();

  static String get projectName {
    final value = envConfigJS?.PROJECT_NAME;
    return (value == null || value.isEmpty) ? _defaultProjectName : value;
  }
}

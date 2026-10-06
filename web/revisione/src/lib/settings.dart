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
}

class AppSettings {
  static const String _defaultApiUrl = 'http://127.0.0.1:5001';
  static const String _defaultProjectName = 'Revisione documenti';

  static String get apiUrl {
    final value = envConfigJS?.REVIEW_API_URL;
    final url = (value == null || value.isEmpty) ? _defaultApiUrl : value;
    return url.endsWith('/') ? url.substring(0, url.length - 1) : url;
  }

  static String get projectName {
    final value = envConfigJS?.PROJECT_NAME;
    return (value == null || value.isEmpty) ? _defaultProjectName : value;
  }
}

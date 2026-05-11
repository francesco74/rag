import 'dart:js_interop';

// 1. Declare the global object (ENV_CONFIG)
// @JS() is used to indicate a global scope definition.
@JS('ENV_CONFIG')
external EnvConfigJS? get envConfigJS;

// 2. Define the structure of the JavaScript object
@JS()
extension type EnvConfigJS._(JSObject _) implements JSObject {
  external String? get DOCUMENT_URL;
  external String? get REST_URL;
  external String? get PROJECT_NAME;
  external String? get API_SECRET_KEY;
  external String? get TOPIC_ID;
}

class AppSettings {
  
  //static const String _productionApiUrl = "http://ia.intranet.provincia.lucca/api";
  //static const String _downloadDocumentUrl = "http://ia.intranet.provincia.lucca/downloads";

  static const String _productionApiUrl = "http://127.0.0.1:5000";
  static const String _downloadDocumentUrl = "http://127.0.0.1:5000";
  static const String _projectName = "BiblioLucca - Progetti innovativi";
  static const String _secretKeyValue = "default_secret_key";
  static const String _topicId = "greenlees";

  static String get apiUrl {
    // Read from JavaScript first (Runtime value)
    final runtimeValue = envConfigJS?.REST_URL;
    
    // If the runtime value is missing or null, fall back to the safe default
    return runtimeValue ?? _productionApiUrl;
  }

  static String get downloadDocumentUrl {
    // Read from JavaScript first (Runtime value)
    final runtimeValue = envConfigJS?.DOCUMENT_URL;
    
    // If the runtime value is missing or null, fall back to the safe default
    return runtimeValue ?? _downloadDocumentUrl;
  }

  static String get projectName {
    // Read from JavaScript first (Runtime value)
    final projectName = envConfigJS?.PROJECT_NAME;

    return projectName ?? _projectName;

  }

   static String get apiSecretKeyValue {
    // Read from JavaScript first (Runtime value)
    final secretKey = envConfigJS?.API_SECRET_KEY;

    return secretKey  ?? _secretKeyValue;

  }

  static String get getTopicId {
    // Read from JavaScript first (Runtime value)
    final topicId = envConfigJS?.TOPIC_ID;

    return topicId  ?? _topicId;

   }
}
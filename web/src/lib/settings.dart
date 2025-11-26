class AppSettings {
  
  //static const String _productionApiUrl = "http://ia.intranet.provincia.lucca/api";
  //static const String _downloadDocumentUrl = "http://ia.intranet.provincia.lucca/downloads";

  static const String _productionApiUrl = "http://127.0.0.1:5000";
  static const String _downloadDocumentUrl = "http://127.0.0.1:5000";
  

  /// Determines the correct API URL for the web app.
  static String get apiUrl {
    // When running in development/debug mode, use the debug URL.
    return _productionApiUrl;
  }

  static String get downloadDocumentUrl {
    // When running in development/debug mode, use the debug URL.
    return _downloadDocumentUrl;
  }
}
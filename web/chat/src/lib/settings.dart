import 'dart:convert';
import 'dart:js_interop';

import 'package:http/http.dart' as http;

/// Definita in web/index.html: nome e icona del sito nella scheda del browser.
@JS('setSiteIdentity')
external void _setSiteIdentity(String name, bool hasIcon);

/// Il sito è deciso dal dominio: il backend, raggiunto sullo stesso indirizzo
/// sotto /api, dice quale archivio, nome e home page gli corrispondono.
class AppSettings {
  static const String _apiOverride = String.fromEnvironment('API_URL');

  static String get apiUrl =>
      _apiOverride.isNotEmpty ? _apiOverride : '${Uri.base.origin}/api';

  static String get downloadDocumentUrl => '$apiUrl/files';

  static String projectName = 'Archivio documentale';
  static String getTopicId = '';
  static bool allowDateFilter = false;
  static Map<String, String> welcomeHtml = const {};
  static List<Map<String, dynamic>> buttons = const [];

  /// false = nessun sito per questo dominio. Un errore di rete solleva
  /// un'eccezione: va distinto, perché lì si può solo riprovare.
  static Future<bool> load() async {
    final res = await http.get(Uri.parse('$apiUrl/site'));
    if (res.statusCode == 404) return false;
    if (res.statusCode != 200) {
      throw StateError('Risposta inattesa: ${res.statusCode}');
    }
    final data = jsonDecode(utf8.decode(res.bodyBytes)) as Map<String, dynamic>;
    projectName = data['name'] as String;
    getTopicId = data['topic_id'] as String;
    allowDateFilter = data['allow_date_filter'] == true;
    welcomeHtml = {
      for (final e in (data['welcome'] as Map<String, dynamic>).entries)
        if (e.value is String) e.key: e.value as String,
    };
    buttons = List<Map<String, dynamic>>.from(data['buttons'] as List? ?? const []);
    _setSiteIdentity(projectName, data['has_icon'] == true);
    return true;
  }
}

import 'dart:convert';

import 'package:flutter/foundation.dart';
import 'package:http/http.dart' as http;

import 'settings.dart';

/// Utente autenticato (stessi utenti della revisione documenti).
class ChatUser {
  ChatUser.fromJson(Map<String, dynamic> j)
      : username = j['username'] as String,
        displayName = (j['display_name'] ?? j['username']) as String;

  final String username;
  final String displayName;
}

/// Esito del controllo su un documento prima di aprirlo.
enum DocumentAccess { allowed, loginRequired, forbidden, notFound, unknown }

class AuthException implements Exception {
  AuthException(this.message);
  final String message;
}

/// Accesso facoltativo alla chat, per consultare i documenti degli archivi
/// riservati. Il token di sessione non passa mai da qui: l'API lo mette in
/// un cookie HttpOnly che il browser invia da solo (anche quando il
/// documento si apre in una nuova scheda) e che JavaScript non può leggere.
class ChatAuth {
  ChatAuth._();
  static final ChatAuth instance = ChatAuth._();

  final ValueNotifier<ChatUser?> user = ValueNotifier(null);

  Uri _auth(String path) => Uri.parse('${AppSettings.apiUrl}/auth/$path');

  /// Riprende la sessione, se il cookie è ancora valido.
  Future<void> restore() async {
    try {
      final res = await http.get(_auth('me'));
      user.value = res.statusCode == 200
          ? ChatUser.fromJson(
              (jsonDecode(utf8.decode(res.bodyBytes)) as Map)['user']
                  as Map<String, dynamic>)
          : null;
    } catch (_) {
      user.value = null;
    }
  }

  Future<void> login(String username, String password) async {
    http.Response res;
    try {
      res = await http.post(
        _auth('login'),
        headers: {'Content-Type': 'application/json'},
        body: jsonEncode({'username': username, 'password': password}),
      );
    } catch (_) {
      throw AuthException('');
    }
    Map<String, dynamic> data = {};
    try {
      data = jsonDecode(utf8.decode(res.bodyBytes)) as Map<String, dynamic>;
    } catch (_) {}
    if (res.statusCode != 200) {
      throw AuthException((data['error'] ?? '').toString());
    }
    user.value = ChatUser.fromJson(data['user'] as Map<String, dynamic>);
  }

  Future<void> logout() async {
    try {
      await http.post(_auth('logout'));
    } catch (_) {
      // Si esce comunque dall'interfaccia.
    } finally {
      user.value = null;
    }
  }

  /// Controlla, senza scaricarlo, se il documento si può aprire.
  Future<DocumentAccess> checkDocument(String url) async {
    try {
      final res = await http.head(Uri.parse(url),
          headers: {'Accept': 'application/json'});
      return switch (res.statusCode) {
        >= 200 && < 400 => DocumentAccess.allowed,
        401 => DocumentAccess.loginRequired,
        403 => DocumentAccess.forbidden,
        404 => DocumentAccess.notFound,
        _ => DocumentAccess.unknown,
      };
    } catch (_) {
      return DocumentAccess.unknown;
    }
  }
}

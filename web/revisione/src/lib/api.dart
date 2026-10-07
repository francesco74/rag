import 'dart:convert';

import 'package:flutter/foundation.dart';
import 'package:http/http.dart' as http;
import 'package:web/web.dart' as web;

import 'settings.dart';

/// Errore restituito dal servizio di revisione.
class ApiException implements Exception {
  ApiException(this.statusCode, this.message, {this.details = const []});

  final int statusCode;
  final String message;
  final List<String> details;

  bool get isUnauthorized => statusCode == 401;
  bool get isForbidden => statusCode == 403;
  bool get isConflict => statusCode == 409;

  @override
  String toString() =>
      details.isEmpty ? message : '$message\n• ${details.join('\n• ')}';
}

/// Permessi concessi dai ruoli (vedi api/src/review_permissions.py).
/// L'interfaccia decide cosa mostrare in base ai permessi, mai ai nomi dei
/// ruoli: il backend li verifica comunque su ogni richiesta.
abstract final class Permission {
  static const read = 'documenti.lettura';
  static const editText = 'documenti.testo';
  static const editMetadata = 'documenti.metadati';
  static const changeStatus = 'documenti.stato';
  static const manageUsers = 'utenti.gestione';
}

class ReviewUser {
  ReviewUser.fromJson(Map<String, dynamic> j)
      : id = j['id'] as int,
        username = j['username'] as String,
        displayName = (j['display_name'] ?? j['username']) as String,
        roles = List<String>.from(j['roles'] as List? ?? const []),
        permissions = Set<String>.from(j['permissions'] as List? ?? const []);

  final int id;
  final String username;
  final String displayName;
  final List<String> roles;
  final Set<String> permissions;

  bool can(String permission) => permissions.contains(permission);
}

class SubTopic {
  SubTopic(this.id, this.description);
  final String id;
  final String description;
}

class Topic {
  Topic(this.id, this.description, this.subTopics);
  final String id;
  final String description;
  final List<SubTopic> subTopics;
}

/// Chiave di un documento: la stessa terna usata dall'ingest.
class DocKey {
  const DocKey(this.source, this.topicId, this.subTopicId);
  final String source;
  final String topicId;
  final String subTopicId;

  Map<String, String> toJson() =>
      {'source': source, 'topic_id': topicId, 'sub_topic_id': subTopicId};
}

class DocumentSummary {
  DocumentSummary.fromJson(Map<String, dynamic> j)
      : key = DocKey(j['source'] as String, j['topic_id'] as String,
            j['sub_topic_id'] as String),
        fileName = j['file_name'] as String?,
        title = (j['title'] ?? j['source']).toString(),
        anno = j['anno']?.toString(),
        numero = j['numero']?.toString(),
        data = j['data']?.toString(),
        indexedAt = j['indexed_at'] as String?,
        status = j['status'] as String,
        statusUpdatedBy = j['status_updated_by'] as String?;

  final DocKey key;
  final String? fileName;
  final String title;
  final String? anno;
  final String? numero;
  final String? data;
  final String? indexedAt;
  final String status;
  final String? statusUpdatedBy;
}

class DocumentPage {
  DocumentPage(this.items, this.total, this.page, this.pageSize);
  final List<DocumentSummary> items;
  final int total;
  final int page;
  final int pageSize;
}

class OriginalFile {
  OriginalFile.fromJson(Map<String, dynamic> j)
      : name = j['name'] as String,
        size = j['size'] as int,
        mimeType = j['mime_type'] as String,
        token = j['token'] as String?;

  final String name;
  final int size;
  final String mimeType;
  final String? token;

  String? url({bool download = false}) {
    if (token == null) return null;
    final q = Uri(queryParameters: {
      'token': token!,
      if (download) 'download': '1',
    }).query;
    return '${AppSettings.apiUrl}/document/file?$q';
  }

  /// Tipi che il browser sa mostrare da solo in un riquadro.
  bool get previewable =>
      mimeType == 'application/pdf' ||
      mimeType.startsWith('image/') ||
      mimeType.startsWith('text/');
}

class ReindexState {
  ReindexState.fromJson(Map<String, dynamic> j)
      : state = j['state'] as String,
        since = j['since'] as String?,
        error = j['error'] as String?;

  final String state; // pending | error
  final String? since;
  final String? error;

  bool get isPending => state == 'pending';
  bool get isError => state == 'error';
}

class ReviewDocument {
  ReviewDocument.fromJson(Map<String, dynamic> j)
      : key = DocKey(j['source'] as String, j['topic_id'] as String,
            j['sub_topic_id'] as String),
        fileName = j['file_name'] as String?,
        parentCount = j['parent_count'] as int,
        indexedAt = j['indexed_at'] as String?,
        metadata = Map<String, dynamic>.from(j['metadata'] as Map),
        protectedKeys = List<String>.from(j['protected_keys'] as List),
        originalFile = j['original_file'] == null
            ? null
            : OriginalFile.fromJson(j['original_file'] as Map<String, dynamic>),
        status = (j['review_status'] as Map)['status'] as String,
        statusNote = (j['review_status'] as Map)['note'] as String?,
        statusUpdatedBy = (j['review_status'] as Map)['updated_by'] as String?,
        reindex = j['reindex'] == null
            ? null
            : ReindexState.fromJson(j['reindex'] as Map<String, dynamic>),
        content = j['content'] as String? ?? '',
        contentOrigin = j['content_origin'] as String?,
        contentHash = j['content_hash'] as String?;

  final DocKey key;
  final String? fileName;
  final int parentCount;
  final String? indexedAt;
  final Map<String, dynamic> metadata;
  final List<String> protectedKeys;
  final OriginalFile? originalFile;
  final String status;
  final String? statusNote;
  final String? statusUpdatedBy;
  final ReindexState? reindex;
  final String content;
  final String? contentOrigin;
  final String? contentHash;

  String get title =>
      (metadata['oggetto'] as String?)?.trim().isNotEmpty == true
          ? metadata['oggetto'] as String
          : (fileName ?? key.source);
}

class HistoryEntry {
  HistoryEntry.fromJson(Map<String, dynamic> j)
      : id = j['id'] as int,
        createdAt = j['created_at'] as String?,
        username = j['username'] as String,
        action = j['action'] as String,
        note = j['note'] as String?,
        details = Map<String, dynamic>.from((j['details'] ?? {}) as Map),
        oldValue = j['old_value'] as String?,
        newValue = j['new_value'] as String?;

  final int id;
  final String? createdAt;
  final String username;
  final String action; // content | metadata | status
  final String? note;
  final Map<String, dynamic> details;
  final String? oldValue;
  final String? newValue;
}

/// Client del servizio di revisione. Il token di sessione vive in
/// sessionStorage: sopravvive al ricaricamento della pagina ma non alla
/// chiusura della scheda.
class ReviewApi {
  ReviewApi._();
  static final ReviewApi instance = ReviewApi._();

  static const _tokenKey = 'review_token';

  final ValueNotifier<ReviewUser?> currentUser = ValueNotifier(null);
  String? _token;

  String? get _storedToken => web.window.sessionStorage.getItem(_tokenKey);

  /// Ripristina la sessione salvata, se ancora valida.
  Future<bool> restoreSession() async {
    _token = _storedToken;
    if (_token == null) return false;
    try {
      final res = await _request('GET', '/auth/me');
      currentUser.value =
          ReviewUser.fromJson(res['user'] as Map<String, dynamic>);
      return true;
    } catch (_) {
      _clearSession();
      return false;
    }
  }

  Future<void> login(String username, String password) async {
    final res = await _request('POST', '/auth/login',
        body: {'username': username, 'password': password}, auth: false);
    _token = res['token'] as String;
    web.window.sessionStorage.setItem(_tokenKey, _token!);
    currentUser.value = ReviewUser.fromJson(res['user'] as Map<String, dynamic>);
  }

  /// Chiude la sessione: il server invalida il token, poi lo si rimuove dal
  /// browser. Anche se il server non risponde l'utente esce comunque.
  Future<void> logout() async {
    try {
      if (_token != null) await _request('POST', '/auth/logout');
    } catch (_) {
      // Server irraggiungibile o token già scaduto: si esce lo stesso.
    } finally {
      _clearSession();
    }
  }

  /// Vero se l'utente corrente ha il permesso indicato.
  bool can(String permission) => currentUser.value?.can(permission) ?? false;

  void _clearSession() {
    _token = null;
    web.window.sessionStorage.removeItem(_tokenKey);
    currentUser.value = null;
  }

  Future<List<Topic>> topics() async {
    final res = await _request('GET', '/topics');
    return (res['topics'] as List).map((t) {
      final m = t as Map<String, dynamic>;
      return Topic(
        m['id'] as String,
        (m['description'] ?? m['id']).toString(),
        (m['sub_topics'] as List)
            .map((s) => SubTopic((s as Map)['id'] as String,
                (s['description'] ?? s['id']).toString()))
            .toList(),
      );
    }).toList();
  }

  Future<DocumentPage> documents({
    String? topicId,
    String? subTopicId,
    String? query,
    String? status,
    int page = 1,
    int pageSize = 25,
  }) async {
    final res = await _request('GET', '/documents', query: {
      if (topicId != null && topicId.isNotEmpty) 'topic_id': topicId,
      if (subTopicId != null && subTopicId.isNotEmpty) 'sub_topic_id': subTopicId,
      if (query != null && query.isNotEmpty) 'q': query,
      if (status != null && status.isNotEmpty) 'status': status,
      'page': '$page',
      'page_size': '$pageSize',
    });
    return DocumentPage(
      (res['items'] as List)
          .map((e) => DocumentSummary.fromJson(e as Map<String, dynamic>))
          .toList(),
      res['total'] as int,
      res['page'] as int,
      res['page_size'] as int,
    );
  }

  Future<ReviewDocument> document(DocKey key) async =>
      ReviewDocument.fromJson(
          await _request('GET', '/document', query: key.toJson()));

  Future<ReviewDocument> saveContent(DocKey key, String content,
      {required String? baseHash, String? note, bool markReviewed = false}) async {
    final res = await _request('PUT', '/document/content', body: {
      ...key.toJson(),
      'content': content,
      'base_hash': baseHash,
      'note': note,
      'mark_reviewed': markReviewed,
    });
    return ReviewDocument.fromJson(res['document'] as Map<String, dynamic>);
  }

  Future<ReviewDocument> saveMetadata(
      DocKey key, Map<String, dynamic> metadata, {String? note}) async {
    final res = await _request('PUT', '/document/metadata',
        body: {...key.toJson(), 'metadata': metadata, 'note': note});
    return ReviewDocument.fromJson(res['document'] as Map<String, dynamic>);
  }

  Future<void> setStatus(DocKey key, String status, {String? note}) =>
      _request('PUT', '/document/status',
          body: {...key.toJson(), 'status': status, 'note': note});

  Future<List<HistoryEntry>> history(DocKey key) async {
    final res =
        await _request('GET', '/document/history', query: key.toJson());
    return (res['items'] as List)
        .map((e) => HistoryEntry.fromJson(e as Map<String, dynamic>))
        .toList();
  }

  Future<HistoryEntry> historyEntry(int id) async => HistoryEntry.fromJson(
      await _request('GET', '/document/history/$id'));

  Future<Map<String, dynamic>> _request(String method, String path,
      {Map<String, String>? query, Object? body, bool auth = true}) async {
    final uri = Uri.parse('${AppSettings.apiUrl}$path')
        .replace(queryParameters: query);
    final headers = <String, String>{
      'Content-Type': 'application/json',
      if (auth && _token != null) 'Authorization': 'Bearer $_token',
    };

    http.Response res;
    try {
      final encoded = body == null ? null : jsonEncode(body);
      res = switch (method) {
        'GET' => await http.get(uri, headers: headers),
        'POST' => await http.post(uri, headers: headers, body: encoded),
        'PUT' => await http.put(uri, headers: headers, body: encoded),
        _ => throw ArgumentError(method),
      };
    } catch (e) {
      throw ApiException(0, 'Servizio di revisione non raggiungibile.');
    }

    Map<String, dynamic> data = {};
    if (res.body.isNotEmpty) {
      try {
        data = jsonDecode(utf8.decode(res.bodyBytes)) as Map<String, dynamic>;
      } catch (_) {}
    }

    if (res.statusCode >= 200 && res.statusCode < 300) return data;

    if (res.statusCode == 401 && auth) {
      // Sessione scaduta o utente disattivato: si torna al login.
      _clearSession();
    }
    throw ApiException(
      res.statusCode,
      (data['error'] ?? 'Errore ${res.statusCode}').toString(),
      details: (data['details'] as List?)?.map((e) => e.toString()).toList() ??
          const [],
    );
  }
}

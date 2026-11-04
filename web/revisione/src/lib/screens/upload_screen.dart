import 'dart:async';
import 'dart:js_interop';

import 'package:flutter/material.dart';
import 'package:web/web.dart' as web;

import '../api.dart';
import '../widgets/common.dart';
import '../widgets/user_menu.dart';

const _accept = '.pdf,.txt,.md,.png,.jpg,.jpeg,.p7m';

/// Un file da inviare e le sottocartelle di origine ('' se scelto da solo).
class _Pick {
  const _Pick(this.file, this.folder);

  final web.File file;
  final String folder;

  String get path => folder.isEmpty ? file.name : '$folder/${file.name}';

  /// Una cartella contiene anche file da non inviare (Thumbs.db, .DS_Store, ...).
  bool get wanted {
    final name = file.name.toLowerCase();
    return !path.split('/').any((part) => part.startsWith('.')) &&
        _accept.split(',').any(name.endsWith);
  }
}

/// Carica i file di un archivio e ne segue lo stato fino all'indicizzazione.
class UploadScreen extends StatefulWidget {
  const UploadScreen({super.key, required this.active});

  /// false mentre è aperta un'altra funzione: niente aggiornamenti né file trascinati.
  final bool active;

  @override
  State<UploadScreen> createState() => _UploadScreenState();
}

class _UploadScreenState extends State<UploadScreen> {
  final _api = ReviewApi.instance;
  List<Topic> _topics = [];
  String? _topicId;
  String? _subTopicId;
  bool _replace = false;
  List<_Pick> _files = [];
  int _ignored = 0;
  bool _sending = false;
  String? _progress;
  final List<String> _failures = [];
  List<Map<String, dynamic>> _items = [];
  // Si apre sugli errori: sono quelli che chiedono un intervento.
  String _filter = 'errori';
  Map<String, int> _counts = {};
  int _indexedDays = 3;
  int _maxMb = 100;
  Object? _loadError;
  Timer? _refresh;
  late final JSFunction _onDragOver = ((web.Event e) => e.preventDefault()).toJS;
  late final JSFunction _onDrop = _handleDrop.toJS;

  @override
  void initState() {
    super.initState();
    web.document.addEventListener('dragover', _onDragOver);
    web.document.addEventListener('drop', _onDrop);
    _load();
    _refresh = Timer.periodic(const Duration(seconds: 8), (_) {
      if (widget.active) _loadItems();
    });
  }

  @override
  void didUpdateWidget(UploadScreen oldWidget) {
    super.didUpdateWidget(oldWidget);
    // Archivi e sottocategorie possono essere cambiati nell'amministrazione.
    if (widget.active && !oldWidget.active) _load();
  }

  @override
  void dispose() {
    _refresh?.cancel();
    web.document.removeEventListener('dragover', _onDragOver);
    web.document.removeEventListener('drop', _onDrop);
    super.dispose();
  }

  Future<void> _load() async {
    try {
      final topics = await _api.topics();
      if (!mounted) return;
      setState(() {
        _topics = topics;
        if (!topics.any((t) => t.id == _topicId)) _topicId = topics.length == 1 ? topics.first.id : null;
        final current = topics.where((t) => t.id == _topicId).expand((t) => t.subTopics);
        if (!current.any((s) => s.id == _subTopicId)) _subTopicId = null;
      });
      await _loadItems();
    } catch (e) {
      if (mounted) setState(() => _loadError = e);
    }
  }

  Future<void> _loadItems() async {
    try {
      final filter = _filter;
      final res = await _api.adminRequest('GET', '/uploads', query: {'status': filter}, admin: false);
      if (!mounted || filter != _filter) return;
      setState(() {
        _items = List<Map<String, dynamic>>.from(res['items'] as List);
        _maxMb = res['max_mb'] as int;
        _counts = Map<String, int>.from(res['counts'] as Map);
        _indexedDays = res['indexed_days'] as int;
      });
    } catch (e) {
      if (mounted && _items.isEmpty) setState(() => _loadError = e);
    }
  }

  void _handleDrop(web.DragEvent event) {
    event.preventDefault();
    final items = event.dataTransfer?.items;
    if (items == null || _sending || !widget.active) return;
    // Le voci trascinate sono leggibili solo durante l'evento: si raccolgono subito.
    final entries = [
      for (var i = 0; i < items.length; i++)
        if (items[i].webkitGetAsEntry() case final entry?) entry,
    ];
    _readEntries(entries);
  }

  Future<void> _readEntries(List<web.FileSystemEntry> entries) async {
    final picks = <_Pick>[];
    try {
      for (final entry in entries) {
        await _walk(entry, '', picks);
      }
    } catch (e) {
      if (mounted) showError(context, 'Lettura della cartella non riuscita: $e');
    }
    if (mounted) _addPicks(picks);
  }

  Future<void> _walk(web.FileSystemEntry entry, String folder, List<_Pick> out) async {
    if (entry.isFile) {
      final done = Completer<web.File>();
      (entry as web.FileSystemFileEntry).file(
          ((web.File f) => done.complete(f)).toJS, ((JSAny e) => done.completeError(e)).toJS);
      out.add(_Pick(await done.future, folder));
      return;
    }
    final inner = folder.isEmpty ? entry.name : '$folder/${entry.name}';
    final reader = (entry as web.FileSystemDirectoryEntry).createReader();
    // readEntries restituisce le voci a blocchi, fino a un blocco vuoto.
    while (true) {
      final done = Completer<List<web.FileSystemEntry>>();
      reader.readEntries(((JSArray<web.FileSystemEntry> batch) => done.complete(batch.toDart)).toJS,
          ((JSAny e) => done.completeError(e)).toJS);
      final batch = await done.future;
      if (batch.isEmpty) return;
      for (final child in batch) {
        await _walk(child, inner, out);
      }
    }
  }

  void _addPicks(List<_Pick> picks) {
    // I file scelti uno a uno passano sempre: se non vanno, il server dice perché.
    final kept = picks.where((p) => p.folder.isEmpty || p.wanted).toList();
    setState(() {
      _files = [
        ..._files,
        for (final pick in kept)
          if (!_files.any((f) => f.path == pick.path)) pick,
      ];
      _ignored += picks.length - kept.length;
      _failures.clear();
    });
  }

  Future<void> _pick() async {
    final input = web.HTMLInputElement()
      ..type = 'file'
      ..multiple = true
      ..accept = _accept;
    input.click();
    await input.onChange.first;
    final list = input.files;
    if (list != null) {
      _addPicks([for (var i = 0; i < list.length; i++) _Pick(list.item(i)!, '')]);
    }
  }

  Future<void> _pickFolder() async {
    final input = web.HTMLInputElement()
      ..type = 'file'
      ..webkitdirectory = true;
    input.click();
    await input.onChange.first;
    final list = input.files;
    if (list == null) return;
    _addPicks([
      for (var i = 0; i < list.length; i++)
        _Pick(list.item(i)!, list.item(i)!.webkitRelativePath.split('/').reversed.skip(1).toList().reversed.join('/')),
    ]);
  }

  Future<void> _send() async {
    final files = _files;
    setState(() {
      _sending = true;
      _failures.clear();
    });
    final failed = <_Pick>[];
    for (final (index, pick) in files.indexed) {
      setState(() => _progress = 'Invio di ${pick.path} (${index + 1} di ${files.length})');
      try {
        final bytes = (await pick.file.arrayBuffer().toDart).toDart.asUint8List();
        await _api.uploadFile(
            topicId: _topicId!,
            subTopicId: _subTopicId!,
            name: pick.file.name,
            folder: pick.folder,
            bytes: bytes,
            replace: _replace);
      } catch (e) {
        failed.add(pick);
        _failures.add('${pick.path}: ${e is ApiException ? e.message : e}');
      }
    }
    if (!mounted) return;
    setState(() {
      _files = failed; // restano da correggere o ritentare
      _ignored = 0;
      _sending = false;
      _progress = null;
      // Si seguono i file appena inviati.
      _filter = 'in_lavorazione';
      _items = [];
    });
    await _loadItems();
  }

  Future<void> _resend(Map<String, dynamic> item) async {
    try {
      await _api.adminRequest('POST', '/uploads/${item['id']}/send', admin: false);
      await _loadItems();
    } catch (e) {
      if (mounted) showError(context, e);
    }
  }

  Future<void> _retry(Map<String, dynamic> item) async {
    try {
      await _api.adminRequest('POST', '/uploads/${item['id']}/retry', admin: false);
      await _loadItems();
    } catch (e) {
      if (mounted) showError(context, e);
    }
  }

  String _size(num bytes) =>
      bytes < 1024 * 1024 ? '${(bytes / 1024).ceil()} KB' : '${(bytes / (1024 * 1024)).toStringAsFixed(1)} MB';

  Widget _form() {
    final subTopics = _topics.where((t) => t.id == _topicId).expand((t) => t.subTopics).toList();
    final total = _files.fold<num>(0, (sum, p) => sum + p.file.size);
    final canSend = !_sending && _files.isNotEmpty && _topicId != null && _subTopicId != null;
    return Card(
      child: Padding(
        padding: const EdgeInsets.all(16),
        child: Column(crossAxisAlignment: CrossAxisAlignment.start, children: [
          Row(children: [
            Expanded(
              child: DropdownButtonFormField<String>(
                key: ValueKey('archivio-$_topicId'),
                initialValue: _topicId,
                decoration: const InputDecoration(labelText: 'Archivio', border: OutlineInputBorder()),
                items: [for (final t in _topics) DropdownMenuItem(value: t.id, child: Text(t.description))],
                onChanged: _sending
                    ? null
                    : (v) => setState(() {
                          _topicId = v;
                          _subTopicId = null;
                        }),
              ),
            ),
            const SizedBox(width: 12),
            Expanded(
              child: DropdownButtonFormField<String>(
                key: ValueKey('serie-$_topicId-$_subTopicId'),
                initialValue: _subTopicId,
                decoration: const InputDecoration(labelText: 'Sottocategoria', border: OutlineInputBorder()),
                items: [for (final s in subTopics) DropdownMenuItem(value: s.id, child: Text(s.description))],
                onChanged: _sending ? null : (v) => setState(() => _subTopicId = v),
              ),
            ),
          ]),
          if (_topicId != null && subTopics.isEmpty)
            const Padding(
              padding: EdgeInsets.only(top: 8),
              child: Text('Questo archivio non ha sottocategorie: un amministratore deve aggiungerle.'),
            ),
          const SizedBox(height: 12),
          Row(children: [
            OutlinedButton.icon(
              onPressed: _sending ? null : _pick,
              icon: const Icon(Icons.attach_file),
              label: const Text('Scegli i file'),
            ),
            const SizedBox(width: 8),
            OutlinedButton.icon(
              onPressed: _sending ? null : _pickFolder,
              icon: const Icon(Icons.folder_open),
              label: const Text('Scegli una cartella'),
            ),
            const SizedBox(width: 12),
            Expanded(
                child: Text('oppure trascinali in questa pagina. Le cartelle conservano le sottocartelle. '
                    'PDF, immagini, testo e .p7m, massimo $_maxMb MB l\'uno.')),
          ]),
          CheckboxListTile(
            contentPadding: EdgeInsets.zero,
            controlAffinity: ListTileControlAffinity.leading,
            title: const Text('Sostituisci i documenti con lo stesso nome'),
            value: _replace,
            onChanged: _sending ? null : (v) => setState(() => _replace = v ?? false),
          ),
          if (_files.isNotEmpty) ...[
            ConstrainedBox(
              constraints: const BoxConstraints(maxHeight: 240),
              child: ListView.builder(
                shrinkWrap: true,
                itemCount: _files.length,
                itemBuilder: (_, i) {
                  final pick = _files[i];
                  return Row(children: [
                    Expanded(child: Text('${pick.path}  (${_size(pick.file.size)})', overflow: TextOverflow.ellipsis)),
                    IconButton(
                      tooltip: 'Togli',
                      icon: const Icon(Icons.close, size: 18),
                      onPressed: _sending ? null : () => setState(() => _files = [..._files]..remove(pick)),
                    ),
                  ]);
                },
              ),
            ),
            const SizedBox(height: 8),
            Text('${_files.length} file, ${_size(total)} in totale. '
                '${_ignored > 0 ? '$_ignored ignorati perché di tipo non ammesso. ' : ''}'
                'L\'ingestione parte subito e può richiedere l\'OCR delle pagine.'),
            const SizedBox(height: 8),
          ],
          for (final failure in _failures)
            Padding(
              padding: const EdgeInsets.only(bottom: 4),
              child: Text(failure, style: TextStyle(color: Theme.of(context).colorScheme.error)),
            ),
          if (_progress != null) ...[
            const LinearProgressIndicator(),
            const SizedBox(height: 4),
            Text(_progress!),
            const SizedBox(height: 8),
          ],
          FilledButton.icon(
            onPressed: canSend ? _send : null,
            icon: const Icon(Icons.upload),
            label: const Text('Carica e avvia l\'ingestione'),
          ),
        ]),
      ),
    );
  }

  Widget _row(Map<String, dynamic> item) {
    final status = item['status'] as String;
    final (label, icon, color) = switch (status) {
      'indicizzato' => ('Indicizzato', Icons.check_circle_outline, Colors.green.shade700),
      'errore' => ('Errore', Icons.error_outline, Theme.of(context).colorScheme.error),
      'da_inviare' => ('Da inviare', Icons.schedule_send, Colors.orange.shade800),
      'in_lavorazione' => ('In lavorazione', Icons.autorenew, Theme.of(context).colorScheme.primary),
      _ => ('In coda', Icons.hourglass_empty, Theme.of(context).colorScheme.outline),
    };
    return Card(
      child: ListTile(
        leading: Icon(icon, color: color),
        title: Text('${item['file_name']}'),
        subtitle: Text('${item['topic_id']} / ${item['sub_topic_id']}  ·  ${_size(item['size_bytes'] as num)}  ·  '
            '${item['uploaded_by']}, ${formatDateTime(item['created_at'] as String?)}'
            '${item['error'] == null ? '' : '\n${item['error']}'}'),
        trailing: switch (status) {
          'da_inviare' => OutlinedButton(onPressed: () => _resend(item), child: const Text('Riprova')),
          'errore' => OutlinedButton(onPressed: () => _retry(item), child: const Text('Riprova')),
          _ => Text(label, style: TextStyle(color: color)),
        },
      ),
    );
  }

  Widget _filters() {
    final options = [
      ('errori', 'Errori'),
      ('in_lavorazione', 'In lavorazione'),
      ('indicizzati', 'Indicizzati (ultimi $_indexedDays giorni)'),
      ('tutti', 'Tutti'),
    ];
    return Wrap(spacing: 8, runSpacing: 8, children: [
      for (final (value, label) in options)
        ChoiceChip(
          label: Text(value == 'tutti' || _counts[value] == null ? label : '$label (${_counts[value]})'),
          selected: _filter == value,
          onSelected: (_) {
            if (_filter == value) return;
            setState(() {
              _filter = value;
              _items = [];
            });
            _loadItems();
          },
        ),
    ]);
  }

  String _emptyText() => switch (_filter) {
        'errori' => 'Nessun file in errore.',
        'in_lavorazione' => 'Nessun file in lavorazione.',
        'indicizzati' => 'Nessun file indicizzato negli ultimi $_indexedDays giorni.',
        _ => 'Nessun file caricato negli ultimi $_indexedDays giorni.',
      };

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        title: const Text('Caricamento documenti'),
        actions: const [UserMenu(), SizedBox(width: 8)],
      ),
      body: _loadError != null
          ? Center(child: Text(_loadError is ApiException ? (_loadError as ApiException).message : '$_loadError'))
          : ListView(padding: const EdgeInsets.all(16), children: [
              _form(),
              const SizedBox(height: 16),
              Text('I miei caricamenti', style: Theme.of(context).textTheme.titleMedium),
              const SizedBox(height: 8),
              _filters(),
              const SizedBox(height: 8),
              if (_items.isEmpty) Text(_emptyText()),
              for (final item in _items) _row(item),
            ]),
    );
  }
}

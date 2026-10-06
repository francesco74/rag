import 'dart:async';

import 'package:flutter/material.dart';

import '../api.dart';
import '../settings.dart';
import '../widgets/common.dart';
import 'document_screen.dart';

class DocumentsScreen extends StatefulWidget {
  const DocumentsScreen({super.key});

  @override
  State<DocumentsScreen> createState() => _DocumentsScreenState();
}

class _DocumentsScreenState extends State<DocumentsScreen> {
  final _api = ReviewApi.instance;
  final _search = TextEditingController();
  Timer? _debounce;

  List<Topic> _topics = [];
  String? _topicId;
  String? _subTopicId;
  String? _status;
  int _page = 1;
  static const _pageSize = 25;

  DocumentPage? _result;
  bool _loading = false;
  String? _error;

  @override
  void initState() {
    super.initState();
    _loadTopics();
    _load();
  }

  @override
  void dispose() {
    _debounce?.cancel();
    _search.dispose();
    super.dispose();
  }

  Future<void> _loadTopics() async {
    try {
      final topics = await _api.topics();
      if (mounted) setState(() => _topics = topics);
    } catch (e) {
      if (mounted) showError(context, e);
    }
  }

  Future<void> _load({int? page}) async {
    setState(() {
      _loading = true;
      _error = null;
      if (page != null) _page = page;
    });
    try {
      final result = await _api.documents(
        topicId: _topicId,
        subTopicId: _subTopicId,
        query: _search.text.trim(),
        status: _status,
        page: _page,
        pageSize: _pageSize,
      );
      if (mounted) setState(() => _result = result);
    } catch (e) {
      if (mounted) setState(() => _error = e.toString());
    } finally {
      if (mounted) setState(() => _loading = false);
    }
  }

  void _onSearchChanged(String _) {
    _debounce?.cancel();
    _debounce = Timer(const Duration(milliseconds: 400), () => _load(page: 1));
  }

  Future<void> _open(DocumentSummary doc) async {
    await Navigator.of(context).push(MaterialPageRoute(
      builder: (_) => DocumentScreen(docKey: doc.key, initialTitle: doc.title),
    ));
    // Al ritorno lo stato o il titolo possono essere cambiati.
    if (mounted) _load();
  }

  List<SubTopic> get _subTopics => _topics
      .firstWhere((t) => t.id == _topicId,
          orElse: () => Topic('', '', const []))
      .subTopics;

  @override
  Widget build(BuildContext context) {
    final user = _api.currentUser.value;
    return Scaffold(
      appBar: AppBar(
        title: Text(AppSettings.projectName),
        actions: [
          if (user != null)
            Padding(
              padding: const EdgeInsets.symmetric(horizontal: 8),
              child: Center(
                child: Row(children: [
                  const Icon(Icons.person_outline, size: 18),
                  const SizedBox(width: 4),
                  Text(user.displayName),
                ]),
              ),
            ),
          IconButton(
            tooltip: 'Esci',
            icon: const Icon(Icons.logout),
            onPressed: _api.logout,
          ),
          const SizedBox(width: 8),
        ],
      ),
      body: Column(children: [
        _filters(),
        const Divider(height: 1),
        if (_loading) const LinearProgressIndicator(minHeight: 2),
        Expanded(child: _list()),
        if (_result != null) _pager(),
      ]),
    );
  }

  Widget _filters() {
    return Padding(
      padding: const EdgeInsets.fromLTRB(16, 12, 16, 12),
      child: Wrap(
        spacing: 12,
        runSpacing: 12,
        crossAxisAlignment: WrapCrossAlignment.center,
        children: [
          SizedBox(
            width: 340,
            child: TextField(
              controller: _search,
              onChanged: _onSearchChanged,
              decoration: InputDecoration(
                prefixIcon: const Icon(Icons.search),
                hintText: 'Cerca per oggetto, nome file, metadati…',
                border: const OutlineInputBorder(),
                isDense: true,
                suffixIcon: _search.text.isEmpty
                    ? null
                    : IconButton(
                        tooltip: 'Cancella ricerca',
                        icon: const Icon(Icons.clear),
                        onPressed: () {
                          _search.clear();
                          _load(page: 1);
                        },
                      ),
              ),
            ),
          ),
          _dropdown<String?>(
            label: 'Archivio',
            value: _topicId,
            items: {null: 'Tutti', for (final t in _topics) t.id: t.description},
            onChanged: (v) {
              setState(() {
                _topicId = v;
                _subTopicId = null;
              });
              _load(page: 1);
            },
          ),
          _dropdown<String?>(
            label: 'Serie',
            value: _subTopicId,
            items: {
              null: 'Tutte',
              for (final s in _subTopics) s.id: s.description
            },
            onChanged: _topicId == null
                ? null
                : (v) {
                    setState(() => _subTopicId = v);
                    _load(page: 1);
                  },
          ),
          _dropdown<String?>(
            label: 'Stato',
            value: _status,
            items: {null: 'Tutti', for (final s in reviewStatuses) s: statusLabel(s)},
            onChanged: (v) {
              setState(() => _status = v);
              _load(page: 1);
            },
          ),
          IconButton(
            tooltip: 'Aggiorna',
            icon: const Icon(Icons.refresh),
            onPressed: _loading ? null : () => _load(),
          ),
        ],
      ),
    );
  }

  Widget _dropdown<T>({
    required String label,
    required T value,
    required Map<T, String> items,
    required ValueChanged<T?>? onChanged,
  }) {
    return SizedBox(
      width: 220,
      child: InputDecorator(
        decoration: InputDecoration(
          labelText: label,
          border: const OutlineInputBorder(),
          isDense: true,
          contentPadding:
              const EdgeInsets.symmetric(horizontal: 12, vertical: 4),
        ),
        child: DropdownButtonHideUnderline(
          child: DropdownButton<T>(
            value: items.containsKey(value) ? value : null,
            isExpanded: true,
            isDense: true,
            onChanged: onChanged,
            items: items.entries
                .map((e) => DropdownMenuItem<T>(
                    value: e.key,
                    child: Text(e.value, overflow: TextOverflow.ellipsis)))
                .toList(),
          ),
        ),
      ),
    );
  }

  Widget _list() {
    final theme = Theme.of(context);
    if (_error != null) {
      return Center(
        child: Column(mainAxisSize: MainAxisSize.min, children: [
          Text(_error!, style: TextStyle(color: theme.colorScheme.error)),
          const SizedBox(height: 8),
          OutlinedButton(onPressed: _load, child: const Text('Riprova')),
        ]),
      );
    }
    final items = _result?.items ?? const [];
    if (items.isEmpty) {
      return Center(
        child: Text(_loading ? '' : 'Nessun documento trovato',
            style: TextStyle(color: theme.colorScheme.onSurfaceVariant)),
      );
    }
    return ListView.separated(
      itemCount: items.length,
      separatorBuilder: (_, _) => const Divider(height: 1),
      itemBuilder: (context, i) {
        final d = items[i];
        final ref = [
          if (d.numero != null && d.numero!.isNotEmpty) 'n. ${d.numero}',
          if (d.anno != null && d.anno!.isNotEmpty) d.anno!,
          if (d.data != null && d.data!.isNotEmpty) d.data!,
        ].join(' · ');
        return ListTile(
          onTap: () => _open(d),
          leading: Icon(_iconFor(d.fileName),
              color: theme.colorScheme.primary),
          title: Text(d.title, maxLines: 2, overflow: TextOverflow.ellipsis),
          subtitle: Text(
            [
              '${d.key.topicId} / ${d.key.subTopicId}',
              if (ref.isNotEmpty) ref,
              d.fileName ?? d.key.source,
            ].join('   ·   '),
            maxLines: 1,
            overflow: TextOverflow.ellipsis,
          ),
          trailing: Column(
            mainAxisAlignment: MainAxisAlignment.center,
            crossAxisAlignment: CrossAxisAlignment.end,
            children: [
              StatusChip(d.status),
              if (d.statusUpdatedBy != null)
                Padding(
                  padding: const EdgeInsets.only(top: 2),
                  child: Text(d.statusUpdatedBy!,
                      style: theme.textTheme.bodySmall),
                ),
            ],
          ),
        );
      },
    );
  }

  IconData _iconFor(String? fileName) {
    final ext = (fileName ?? '').split('.').last.toLowerCase();
    return switch (ext) {
      'pdf' => Icons.picture_as_pdf_outlined,
      'jpg' || 'jpeg' || 'png' || 'tif' || 'tiff' || 'bmp' || 'gif' || 'webp' =>
        Icons.image_outlined,
      _ => Icons.description_outlined,
    };
  }

  Widget _pager() {
    final r = _result!;
    final pages = (r.total / r.pageSize).ceil().clamp(1, 1 << 30);
    final from = r.total == 0 ? 0 : (r.page - 1) * r.pageSize + 1;
    final to = (r.page * r.pageSize).clamp(0, r.total);
    return Material(
      elevation: 2,
      child: Padding(
        padding: const EdgeInsets.symmetric(horizontal: 16, vertical: 4),
        child: Row(children: [
          Text('$from–$to di ${r.total} documenti'),
          const Spacer(),
          IconButton(
            tooltip: 'Pagina precedente',
            icon: const Icon(Icons.chevron_left),
            onPressed: r.page > 1 && !_loading ? () => _load(page: r.page - 1) : null,
          ),
          Text('Pagina ${r.page} di $pages'),
          IconButton(
            tooltip: 'Pagina successiva',
            icon: const Icon(Icons.chevron_right),
            onPressed:
                r.page < pages && !_loading ? () => _load(page: r.page + 1) : null,
          ),
        ]),
      ),
    );
  }
}

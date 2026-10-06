import 'dart:async';

import 'package:flutter/material.dart';

import '../api.dart';
import '../widgets/common.dart';
import '../widgets/history_panel.dart';
import '../widgets/metadata_editor.dart';
import '../widgets/original_viewer.dart';
import '../widgets/text_editor.dart';

class DocumentScreen extends StatefulWidget {
  const DocumentScreen({super.key, required this.docKey, this.initialTitle});

  final DocKey docKey;
  final String? initialTitle;

  @override
  State<DocumentScreen> createState() => _DocumentScreenState();
}

class _DocumentScreenState extends State<DocumentScreen> {
  final _api = ReviewApi.instance;
  final _text = TextEditingController();

  ReviewDocument? _doc;
  String? _loadError;
  bool _savingText = false;
  bool _markReviewed = true;
  bool _metaDirty = false;
  int _historyToken = 0;
  Timer? _poll;

  bool get _textDirty => _doc != null && _text.text != _doc!.content;
  bool get _dirty => _textDirty || _metaDirty;

  @override
  void initState() {
    super.initState();
    _text.addListener(_onTextEdited);
    _load();
  }

  @override
  void dispose() {
    _poll?.cancel();
    _text.dispose();
    super.dispose();
  }

  bool _wasTextDirty = false;
  void _onTextEdited() {
    // Si ricostruisce solo quando cambia lo stato "modificato", non a ogni tasto.
    if (_textDirty != _wasTextDirty) {
      setState(() => _wasTextDirty = _textDirty);
    }
  }

  Future<void> _load({bool keepText = false}) async {
    try {
      final doc = await _api.document(widget.docKey);
      if (!mounted) return;
      setState(() {
        final replaceText = !keepText || !_textDirty;
        _doc = doc;
        _loadError = null;
        if (replaceText) _text.text = doc.content;
        _wasTextDirty = _textDirty;
      });
      _schedulePoll();
    } catch (e) {
      if (mounted) setState(() => _loadError = e.toString());
    }
  }

  /// Durante la re-indicizzazione si ricontrolla periodicamente lo stato,
  /// così il banner si aggiorna da solo quando l'ingest ha finito.
  void _schedulePoll() {
    _poll?.cancel();
    if (_doc?.reindex?.isPending == true) {
      _poll = Timer(const Duration(seconds: 6), () async {
        await _load(keepText: true);
        if (mounted && _doc?.reindex == null) {
          showInfo(context, 'Re-indicizzazione completata: la chat usa già il testo corretto.');
          setState(() => _historyToken++);
        }
      });
    }
  }

  Future<void> _saveText() async {
    final doc = _doc!;
    final note = await askNote(
      context,
      title: 'Salva e re-indicizza',
      confirmLabel: 'Salva',
      message:
          'Il testo corretto sostituirà quello attuale e il documento verrà '
          're-indicizzato (di solito bastano pochi secondi o minuti).',
    );
    if (note == null || !mounted) return;
    setState(() => _savingText = true);
    try {
      final updated = await _api.saveContent(
        doc.key,
        _text.text,
        baseHash: doc.contentHash,
        note: note,
        markReviewed: _markReviewed,
      );
      if (!mounted) return;
      setState(() {
        _doc = updated;
        _text.text = updated.content;
        _wasTextDirty = false;
        _historyToken++;
      });
      _schedulePoll();
      showInfo(context, 'Testo salvato: re-indicizzazione avviata.');
    } on ApiException catch (e) {
      if (!mounted) return;
      if (e.isConflict) {
        await _showConflict(e.message);
      } else {
        showError(context, e);
      }
    } finally {
      if (mounted) setState(() => _savingText = false);
    }
  }

  Future<void> _showConflict(String message) async {
    final reload = await showDialog<bool>(
      context: context,
      builder: (ctx) => AlertDialog(
        title: const Text('Documento modificato da un altro utente'),
        content: Text('$message\n\nLe tue modifiche non sono state salvate. '
            'Puoi copiarle prima di ricaricare.'),
        actions: [
          TextButton(
              onPressed: () => Navigator.pop(ctx, false),
              child: const Text('Resta qui')),
          FilledButton(
              onPressed: () => Navigator.pop(ctx, true),
              child: const Text('Ricarica il documento')),
        ],
      ),
    );
    if (reload == true) await _load();
  }

  Future<bool> _saveMetadata(Map<String, dynamic> meta) async {
    final note = await askNote(context,
        title: 'Salva metadati', confirmLabel: 'Salva');
    if (note == null || !mounted) return false;
    try {
      final updated = await _api.saveMetadata(_doc!.key, meta, note: note);
      if (!mounted) return true;
      setState(() {
        _doc = updated;
        _historyToken++;
      });
      showInfo(context, 'Metadati salvati.');
      return true;
    } catch (e) {
      if (mounted) showError(context, e);
      return false;
    }
  }

  Future<void> _changeStatus(String status) async {
    try {
      await _api.setStatus(_doc!.key, status);
      await _load(keepText: true);
      if (mounted) setState(() => _historyToken++);
    } catch (e) {
      if (mounted) showError(context, e);
    }
  }

  Future<bool> _confirmDiscard() async {
    final ok = await showDialog<bool>(
      context: context,
      builder: (ctx) => AlertDialog(
        title: const Text('Modifiche non salvate'),
        content: const Text('Uscendo perderai le modifiche non salvate.'),
        actions: [
          TextButton(
              onPressed: () => Navigator.pop(ctx, false),
              child: const Text('Resta')),
          FilledButton(
              onPressed: () => Navigator.pop(ctx, true),
              child: const Text('Esci senza salvare')),
        ],
      ),
    );
    return ok == true;
  }

  @override
  Widget build(BuildContext context) {
    final doc = _doc;
    return PopScope(
      canPop: !_dirty,
      onPopInvokedWithResult: (didPop, _) async {
        if (didPop) return;
        if (await _confirmDiscard() && context.mounted) {
          setState(() {
            _text.text = _doc?.content ?? '';
            _metaDirty = false;
          });
          Navigator.of(context).pop();
        }
      },
      child: Scaffold(
        appBar: AppBar(
          title: Column(
            crossAxisAlignment: CrossAxisAlignment.start,
            children: [
              Text(doc?.title ?? widget.initialTitle ?? 'Documento',
                  overflow: TextOverflow.ellipsis),
              Text(
                '${widget.docKey.topicId} / ${widget.docKey.subTopicId}  ·  ${widget.docKey.source}',
                overflow: TextOverflow.ellipsis,
                style: Theme.of(context).textTheme.bodySmall,
              ),
            ],
          ),
          actions: [
            if (doc != null) _statusMenu(doc),
            IconButton(
              tooltip: 'Ricarica',
              icon: const Icon(Icons.refresh),
              onPressed: () async {
                if (_dirty && !await _confirmDiscard()) return;
                _load();
              },
            ),
            const SizedBox(width: 8),
          ],
        ),
        body: _body(),
      ),
    );
  }

  Widget _statusMenu(ReviewDocument doc) {
    return PopupMenuButton<String>(
      tooltip: 'Cambia stato di revisione',
      onSelected: _changeStatus,
      itemBuilder: (_) => [
        for (final s in reviewStatuses)
          CheckedPopupMenuItem(
              value: s, checked: s == doc.status, child: Text(statusLabel(s))),
      ],
      child: Padding(
        padding: const EdgeInsets.symmetric(horizontal: 8),
        child: Row(children: [
          StatusChip(doc.status),
          const Icon(Icons.arrow_drop_down),
        ]),
      ),
    );
  }

  Widget _body() {
    if (_loadError != null && _doc == null) {
      return Center(
        child: Column(mainAxisSize: MainAxisSize.min, children: [
          Text(_loadError!,
              style: TextStyle(color: Theme.of(context).colorScheme.error)),
          const SizedBox(height: 8),
          OutlinedButton(onPressed: _load, child: const Text('Riprova')),
        ]),
      );
    }
    final doc = _doc;
    if (doc == null) return const Center(child: CircularProgressIndicator());

    return LayoutBuilder(builder: (context, constraints) {
      final wide = constraints.maxWidth >= 1100;
      final tabs = _tabs(doc, includeOriginal: !wide);
      return Column(children: [
        ..._banners(doc),
        Expanded(
          child: wide
              ? Row(children: [
                  Expanded(child: OriginalViewer(file: doc.originalFile)),
                  const VerticalDivider(width: 1),
                  Expanded(child: tabs),
                ])
              : tabs,
        ),
      ]);
    });
  }

  List<Widget> _banners(ReviewDocument doc) {
    final theme = Theme.of(context);
    final out = <Widget>[];
    final r = doc.reindex;
    if (r != null && r.isPending) {
      out.add(_banner(
        theme.colorScheme.secondaryContainer,
        const SizedBox(
            width: 16, height: 16, child: CircularProgressIndicator(strokeWidth: 2)),
        'Re-indicizzazione in corso (correzione del ${formatDateTime(r.since)}). '
        'Il testo mostrato è quello corretto; i metadati si potranno modificare al termine.',
      ));
    } else if (r != null && r.isError) {
      out.add(_banner(
        theme.colorScheme.errorContainer,
        Icon(Icons.error_outline, color: theme.colorScheme.onErrorContainer),
        'La re-indicizzazione è fallita: ${r.error ?? 'errore sconosciuto'}. '
        'Il testo corretto è conservato: puoi salvarlo di nuovo per riprovare.',
      ));
    }
    if (doc.contentOrigin == 'parents') {
      out.add(_banner(
        theme.colorScheme.tertiaryContainer,
        Icon(Icons.info_outline, color: theme.colorScheme.onTertiaryContainer),
        'Il testo originale dell\'OCR non è in archivio: quello mostrato è '
        'ricomposto dai frammenti indicizzati e può avere piccole differenze di spaziatura.',
      ));
    }
    return out;
  }

  Widget _banner(Color color, Widget leading, String text) => Material(
        color: color,
        child: Padding(
          padding: const EdgeInsets.symmetric(horizontal: 16, vertical: 10),
          child: Row(children: [
            leading,
            const SizedBox(width: 12),
            Expanded(child: Text(text)),
          ]),
        ),
      );

  Widget _tabs(ReviewDocument doc, {required bool includeOriginal}) {
    final locked = doc.reindex?.isPending == true
        ? 'Re-indicizzazione in corso: i metadati saranno modificabili al termine.'
        : null;
    return DefaultTabController(
      length: includeOriginal ? 4 : 3,
      child: Column(children: [
        TabBar(tabs: [
          if (includeOriginal) const Tab(text: 'Originale'),
          Tab(text: _textDirty ? 'Testo •' : 'Testo'),
          Tab(text: _metaDirty ? 'Metadati •' : 'Metadati'),
          const Tab(text: 'Storico'),
        ]),
        Expanded(
          // IndexedStack e non TabBarView: le schede restano vive, così le
          // modifiche non salvate ai metadati non si perdono cambiando scheda.
          child: Builder(builder: (context) {
            final controller = DefaultTabController.of(context);
            return AnimatedBuilder(
              animation: controller,
              builder: (_, _) => IndexedStack(
                index: controller.index,
                children: [
                  if (includeOriginal) OriginalViewer(file: doc.originalFile),
                  _textTab(),
                  MetadataEditor(
                    metadata: doc.metadata,
                    protectedKeys: doc.protectedKeys,
                    lockedReason: locked,
                    onSave: _saveMetadata,
                    onDirtyChanged: (v) => setState(() => _metaDirty = v),
                  ),
                  HistoryPanel(docKey: doc.key, refreshToken: _historyToken),
                ],
              ),
            );
          }),
        ),
      ]),
    );
  }

  Widget _textTab() {
    final theme = Theme.of(context);
    return Column(children: [
      Expanded(child: OcrTextEditor(controller: _text)),
      const Divider(height: 1),
      Padding(
        padding: const EdgeInsets.symmetric(horizontal: 12, vertical: 8),
        child: Wrap(
          alignment: WrapAlignment.end,
          crossAxisAlignment: WrapCrossAlignment.center,
          spacing: 8,
          runSpacing: 8,
          children: [
            if (_textDirty)
              Text('Modifiche non salvate',
                  style: TextStyle(color: theme.colorScheme.tertiary)),
            Row(mainAxisSize: MainAxisSize.min, children: [
              Checkbox(
                value: _markReviewed,
                onChanged: (v) => setState(() => _markReviewed = v ?? false),
              ),
              const Text('Segna come revisionato'),
            ]),
            TextButton(
              onPressed: _textDirty && !_savingText
                  ? () => setState(() => _text.text = _doc!.content)
                  : null,
              child: const Text('Annulla modifiche'),
            ),
            FilledButton.icon(
              onPressed: _textDirty && !_savingText ? _saveText : null,
              icon: _savingText
                  ? const SizedBox(
                      width: 16,
                      height: 16,
                      child: CircularProgressIndicator(strokeWidth: 2))
                  : const Icon(Icons.save_outlined),
              label: const Text('Salva e re-indicizza'),
            ),
          ],
        ),
      ),
    ]);
  }
}

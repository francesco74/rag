import 'dart:async';
import 'dart:js_interop';

import 'package:flutter/material.dart';
import 'package:web/web.dart' as web;

import '../api.dart';
import '../widgets/common.dart';
import '../widgets/history_panel.dart';
import '../widgets/metadata_editor.dart';
import '../widgets/original_viewer.dart';
import '../widgets/text_editor.dart';
import '../widgets/user_menu.dart';

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

  // Blocco in modifica: chi può modificare apre il documento bloccandolo per
  // gli altri e lo rinnova finché resta qui; gli altri lo vedono in sola
  // lettura. Il backend rifiuta comunque i salvataggi senza blocco.
  DocLock? _lock;
  Timer? _lockTimer;
  late final JSFunction _onPageHide =
      ((web.Event _) => _releaseOnUnload()).toJS;

  bool get _mayEdit => Permission.editing.any(_api.can);
  bool get _haveLock => _lock?.heldByMe == true;

  /// Blocco di un altro utente, se il documento è in revisione da lui.
  DocLock? get _otherLock => _lock != null && !_lock!.heldByMe ? _lock : null;

  // Le funzioni non concesse dai ruoli restano visibili in sola lettura;
  // il backend rifiuta comunque le richieste senza il permesso.
  bool get _canEditText => _api.can(Permission.editText) && _haveLock;
  bool get _canChangeStatus => _api.can(Permission.changeStatus) && _haveLock;

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
    _lockTimer?.cancel();
    if (_lockTimer != null) {
      web.window.removeEventListener('pagehide', _onPageHide);
    }
    if (_haveLock) {
      // Uscita dal documento: lo si libera subito per gli altri.
      unawaited(_api.releaseLock(widget.docKey).catchError((_) {}));
    }
    _text.dispose();
    super.dispose();
  }

  void _releaseOnUnload() {
    if (_haveLock) _api.releaseLockOnUnload(widget.docKey);
  }

  /// Prende il blocco all'apertura e lo rinnova ogni terzo della sua durata.
  void _startLocking(int ttlSeconds) {
    final every = Duration(seconds: (ttlSeconds ~/ 3).clamp(20, 600));
    _lockTimer = Timer.periodic(every, (_) => _refreshLock());
    web.window.addEventListener('pagehide', _onPageHide);
    _refreshLock();
  }

  Future<void> _refreshLock() async {
    final before = _lock;
    final DocLock? now;
    try {
      now = await _api.acquireLock(widget.docKey);
    } catch (_) {
      // Rete o servizio momentaneamente giù: si riprova al prossimo giro,
      // il blocco dura più di un intervallo di rinnovo.
      return;
    }
    if (!mounted) return;
    setState(() => _lock = now);
    final had = before?.heldByMe == true;
    final has = now?.heldByMe == true;
    if (before != null && !had && has) {
      // Chi lo revisionava ha finito: si ricarica la sua versione.
      await _load();
      if (mounted) showInfo(context, 'Il documento è ora libero: puoi modificarlo.');
    } else if (had && !has) {
      _warnLockLost(now);
    }
  }

  void _warnLockLost(DocLock? holder) {
    showError(
      context,
      'Il documento è ora in revisione da ${holder?.displayName ?? 'un altro utente'}: '
      'le modifiche non salvate restano visibili ma non si possono più salvare.',
    );
  }

  /// Un salvataggio è stato rifiutato perché il blocco è di un altro.
  void _onLocked(ApiException e) {
    setState(() => _lock = e.lock);
    _warnLockLost(e.lock);
  }

  Future<void> _forceRelease(DocLock holder) async {
    final ok = await showDialog<bool>(
      context: context,
      builder: (ctx) => AlertDialog(
        title: const Text('Liberare il documento?'),
        content: Text(
            '${holder.displayName} non potrà più salvare le modifiche in corso. '
            'Fallo solo se il blocco è rimasto appeso (ad esempio una scheda '
            'dimenticata aperta).'),
        actions: [
          TextButton(
              onPressed: () => Navigator.pop(ctx, false),
              child: const Text('Annulla')),
          FilledButton(
              onPressed: () => Navigator.pop(ctx, true),
              child: const Text('Libera')),
        ],
      ),
    );
    if (ok != true || !mounted) return;
    try {
      await _api.releaseLock(widget.docKey, force: true);
      await _refreshLock();
    } catch (e) {
      if (mounted) showError(context, e);
    }
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
        _lock = doc.lock;
        _loadError = null;
        if (replaceText) _text.text = doc.content;
        _wasTextDirty = _textDirty;
      });
      _schedulePoll();
      if (_mayEdit && _lockTimer == null) _startLocking(doc.lockTtlSeconds);
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
        markReviewed: _markReviewed && _canChangeStatus,
      );
      if (!mounted) return;
      setState(() {
        _doc = updated;
        _lock = updated.lock;
        _text.text = updated.content;
        _wasTextDirty = false;
        _historyToken++;
      });
      _schedulePoll();
      showInfo(context, 'Testo salvato: re-indicizzazione avviata.');
    } on ApiException catch (e) {
      if (!mounted) return;
      if (e.isLocked) {
        _onLocked(e);
      } else if (e.isConflict) {
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
      final updated = await _api.saveMetadata(_doc!.key, meta,
          baseHash: _doc!.metadataHash, note: note);
      if (!mounted) return true;
      setState(() {
        _doc = updated;
        _lock = updated.lock;
        _historyToken++;
      });
      showInfo(context, 'Metadati salvati.');
      return true;
    } on ApiException catch (e) {
      if (!mounted) return false;
      if (e.isLocked) {
        _onLocked(e);
      } else if (e.isConflict) {
        await _showConflict(e.message);
      } else {
        showError(context, e);
      }
      return false;
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
    } on ApiException catch (e) {
      if (!mounted) return;
      e.isLocked ? _onLocked(e) : showError(context, e);
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
            UserMenu(confirmLogout: () async => !_dirty || await _confirmDiscard()),
            const SizedBox(width: 8),
          ],
        ),
        body: _body(),
      ),
    );
  }

  /// Perché una funzione concessa dal profilo è comunque bloccata.
  String _lockedMessage() {
    final other = _otherLock;
    if (other != null) return 'Il documento è in revisione da ${other.displayName}.';
    return 'Apertura del documento in modifica in corso…';
  }

  Widget _statusMenu(ReviewDocument doc) {
    if (!_canChangeStatus) {
      return Tooltip(
        message: _api.can(Permission.changeStatus)
            ? _lockedMessage()
            : 'Il tuo profilo non permette di cambiare lo stato',
        child: Padding(
          padding: const EdgeInsets.symmetric(horizontal: 8),
          child: StatusChip(doc.status),
        ),
      );
    }
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
    final other = _otherLock;
    if (other != null) {
      out.add(_banner(
        theme.colorScheme.tertiaryContainer,
        Icon(Icons.lock_person_outlined, color: theme.colorScheme.onTertiaryContainer),
        'In revisione da ${other.displayName} dal ${formatDateTime(other.since)}: '
        'puoi solo consultarlo.'
        '${_mayEdit ? ' Diventerà modificabile da solo quando avrà finito.' : ''}',
        action: _api.can(Permission.manageUsers)
            ? TextButton(
                onPressed: () => _forceRelease(other),
                child: const Text('Libera il documento'))
            : null,
      ));
    }
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

  Widget _banner(Color color, Widget leading, String text, {Widget? action}) => Material(
        color: color,
        child: Padding(
          padding: const EdgeInsets.symmetric(horizontal: 16, vertical: 10),
          child: Row(children: [
            leading,
            const SizedBox(width: 12),
            Expanded(child: Text(text)),
            if (action != null) ...[const SizedBox(width: 12), action],
          ]),
        ),
      );

  Widget _tabs(ReviewDocument doc, {required bool includeOriginal}) {
    final locked = !_api.can(Permission.editMetadata)
        ? 'Il tuo profilo non permette di modificare i metadati.'
        : !_haveLock
            ? _lockedMessage()
            : doc.reindex?.isPending == true
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
      Expanded(
          child: OcrTextEditor(controller: _text, readOnly: !_canEditText)),
      const Divider(height: 1),
      if (!_canEditText)
        Padding(
          padding: const EdgeInsets.symmetric(horizontal: 12, vertical: 12),
          child: Row(children: [
            Icon(Icons.lock_outline, size: 18, color: theme.colorScheme.outline),
            const SizedBox(width: 8),
            Expanded(
                child: Text(_api.can(Permission.editText)
                    ? 'Sola lettura. ${_lockedMessage()}'
                    : 'Sola lettura: il tuo profilo non permette di correggere il testo.')),
          ]),
        )
      else
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
            if (_canChangeStatus)
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

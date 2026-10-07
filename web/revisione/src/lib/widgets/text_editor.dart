import 'package:flutter/material.dart';
import 'package:flutter/services.dart';

import 'formatted_editor.dart';
import 'markdown_view.dart';

/// Editor del testo OCR con barra Trova/Sostituisci: gli errori di
/// riconoscimento tendono a ripetersi (stessa parola sbagliata in più punti),
/// quindi "sostituisci tutto" fa risparmiare molto tempo.
///
/// Se il testo è in Markdown si apre nella vista "Formattato" (titoli,
/// tabelle, elenchi formattati), modificabile un blocco alla volta; la vista
/// "Sorgente" mostra il Markdown grezzo e serve per Trova e sostituisci.
class OcrTextEditor extends StatefulWidget {
  const OcrTextEditor({
    super.key,
    required this.controller,
    this.readOnly = false,
  });

  final TextEditingController controller;
  final bool readOnly;

  @override
  State<OcrTextEditor> createState() => _OcrTextEditorState();
}

class _OcrTextEditorState extends State<OcrTextEditor> {
  final _textFocus = FocusNode();
  final _findFocus = FocusNode();
  final _find = TextEditingController();
  final _replace = TextEditingController();
  bool _showFind = false;
  bool _matchCase = false;
  List<int> _matches = const [];
  int _current = -1;
  late bool _isMarkdown = looksLikeMarkdown(widget.controller.text);
  late bool _readable = _isMarkdown;

  @override
  void initState() {
    super.initState();
    widget.controller.addListener(_onTextChanged);
  }

  @override
  void didUpdateWidget(covariant OcrTextEditor oldWidget) {
    super.didUpdateWidget(oldWidget);
    if (oldWidget.controller != widget.controller) {
      oldWidget.controller.removeListener(_onTextChanged);
      widget.controller.addListener(_onTextChanged);
    }
  }

  @override
  void dispose() {
    widget.controller.removeListener(_onTextChanged);
    _textFocus.dispose();
    _findFocus.dispose();
    _find.dispose();
    _replace.dispose();
    super.dispose();
  }

  String? _lastText;
  void _onTextChanged() {
    final text = widget.controller.text;
    // Un documento riconosciuto come Markdown resta tale: la vista non deve
    // cambiare sotto le mani di chi sta modificando (es. cancellando l'unico
    // titolo). Un testo grezzo che diventa Markdown offre la vista
    // formattata ma non ci passa da solo.
    if (!_isMarkdown && text != _lastText && looksLikeMarkdown(text)) {
      setState(() => _isMarkdown = true);
    }
    if (_showFind && text != _lastText) {
      _lastText = text;
      _computeMatches(keepPosition: true);
    }
    _lastText = text;
  }

  void _computeMatches({bool keepPosition = false}) {
    final needle = _find.text;
    final result = <int>[];
    if (needle.isNotEmpty) {
      final hay = _matchCase
          ? widget.controller.text
          : widget.controller.text.toLowerCase();
      final n = _matchCase ? needle : needle.toLowerCase();
      var i = hay.indexOf(n);
      while (i >= 0) {
        result.add(i);
        i = hay.indexOf(n, i + n.length);
      }
    }
    setState(() {
      _matches = result;
      if (!keepPosition || _current >= result.length) {
        _current = result.isEmpty ? -1 : 0;
      }
    });
  }

  void _select(int index, {bool focusText = true}) {
    if (_matches.isEmpty) return;
    final i = index % _matches.length;
    final start = _matches[i];
    setState(() => _current = i);
    widget.controller.selection =
        TextSelection(baseOffset: start, extentOffset: start + _find.text.length);
    // Il focus sul testo fa scorrere l'editor fino alla selezione.
    if (focusText) _textFocus.requestFocus();
  }

  bool get _currentIsSelected {
    if (_current < 0 || _current >= _matches.length) return false;
    final sel = widget.controller.selection;
    return sel.start == _matches[_current] &&
        sel.end == _matches[_current] + _find.text.length;
  }

  void _replaceCurrent() {
    if (widget.readOnly || _matches.isEmpty) return;
    if (!_currentIsSelected) {
      _select(_current < 0 ? 0 : _current);
      return;
    }
    final start = _matches[_current];
    final text = widget.controller.text;
    final updated = text.replaceRange(start, start + _find.text.length, _replace.text);
    final next = _current;
    widget.controller.value = TextEditingValue(
      text: updated,
      selection: TextSelection.collapsed(offset: start + _replace.text.length),
    );
    _computeMatches(keepPosition: true);
    if (_matches.isNotEmpty) _select(next.clamp(0, _matches.length - 1));
  }

  void _replaceAll() {
    if (widget.readOnly || _matches.isEmpty) return;
    final count = _matches.length;
    final buffer = StringBuffer();
    final text = widget.controller.text;
    var last = 0;
    for (final m in _matches) {
      buffer
        ..write(text.substring(last, m))
        ..write(_replace.text);
      last = m + _find.text.length;
    }
    buffer.write(text.substring(last));
    widget.controller.value = TextEditingValue(
      text: buffer.toString(),
      selection: const TextSelection.collapsed(offset: 0),
    );
    _computeMatches();
    ScaffoldMessenger.of(context).showSnackBar(
        SnackBar(content: Text('$count occorrenze sostituite')));
  }

  void _toggleFind([bool? show]) {
    setState(() {
      _showFind = show ?? !_showFind;
      // Trova e sostituisci lavora sul testo sorgente.
      if (_showFind) _readable = false;
    });
    if (_showFind) {
      final sel = widget.controller.selection;
      if (sel.isValid && !sel.isCollapsed && sel.end - sel.start < 100) {
        _find.text = sel.textInside(widget.controller.text);
      }
      _lastText = widget.controller.text;
      _computeMatches();
      _findFocus.requestFocus();
    } else {
      _textFocus.requestFocus();
    }
  }

  @override
  Widget build(BuildContext context) {
    final theme = Theme.of(context);
    return CallbackShortcuts(
      bindings: {
        const SingleActivator(LogicalKeyboardKey.keyF, control: true): () =>
            _toggleFind(true),
        const SingleActivator(LogicalKeyboardKey.keyF, meta: true): () =>
            _toggleFind(true),
        const SingleActivator(LogicalKeyboardKey.keyH, control: true): () =>
            _toggleFind(true),
        const SingleActivator(LogicalKeyboardKey.escape): () {
          if (_showFind) _toggleFind(false);
        },
      },
      child: Column(children: [
        Material(
          color: theme.colorScheme.surfaceContainer,
          child: Padding(
            padding: const EdgeInsets.symmetric(horizontal: 8, vertical: 2),
            // Nel pannello affiancato all'originale lo spazio è poco: sotto
            // una certa larghezza i pulsanti mostrano solo l'icona.
            child: LayoutBuilder(builder: (context, constraints) {
              final compact = constraints.maxWidth < 700;
              final tiny = constraints.maxWidth < 420;
              return Row(children: [
                compact
                    ? IconButton(
                        tooltip: 'Trova e sostituisci',
                        onPressed: () => _toggleFind(),
                        icon: const Icon(Icons.find_replace, size: 20),
                      )
                    : TextButton.icon(
                        onPressed: () => _toggleFind(),
                        icon: const Icon(Icons.find_replace, size: 18),
                        label: const Text('Trova e sostituisci'),
                      ),
                if (_isMarkdown) ...[
                  const SizedBox(width: 8),
                  _viewSelector(showLabels: !tiny),
                ],
                const SizedBox(width: 8),
                Expanded(
                  child: ListenableBuilder(
                    listenable: widget.controller,
                    builder: (_, _) {
                      final text = widget.controller.text;
                      final words = RegExp(r'\S+').allMatches(text).length;
                      return Text(
                        compact
                            ? '$words parole'
                            : '$words parole · ${text.length} caratteri',
                        style: theme.textTheme.bodySmall,
                        overflow: TextOverflow.ellipsis,
                        textAlign: TextAlign.end,
                      );
                    },
                  ),
                ),
              ]);
            }),
          ),
        ),
        if (_showFind) _findBar(theme),
        if (_readable && !widget.readOnly) _formattedHint(theme),
        const Divider(height: 1),
        Expanded(
          child: _readable
              ? FormattedEditor(
                  controller: widget.controller,
                  readOnly: widget.readOnly,
                )
              : TextField(
            controller: widget.controller,
            focusNode: _textFocus,
            readOnly: widget.readOnly,
            expands: true,
            maxLines: null,
            minLines: null,
            keyboardType: TextInputType.multiline,
            textAlignVertical: TextAlignVertical.top,
            style: const TextStyle(
                fontFamily: 'monospace', fontSize: 14, height: 1.45),
            decoration: const InputDecoration(
              border: InputBorder.none,
              contentPadding: EdgeInsets.all(16),
            ),
          ),
        ),
      ]),
    );
  }

  /// Selettore tra vista formattata e sorgente Markdown.
  Widget _viewSelector({bool showLabels = true}) {
    return SegmentedButton<bool>(
      showSelectedIcon: false,
      style: const ButtonStyle(visualDensity: VisualDensity.compact),
      segments: [
        ButtonSegment(
          value: true,
          icon: const Icon(Icons.article_outlined, size: 18),
          label: showLabels ? const Text('Formattato') : null,
          tooltip: widget.readOnly
              ? 'Testo formattato, più facile da leggere'
              : 'Testo formattato: clicca un paragrafo o una tabella per modificarlo',
        ),
        ButtonSegment(
          value: false,
          icon: const Icon(Icons.code, size: 18),
          label: showLabels ? const Text('Sorgente') : null,
          tooltip: 'Testo Markdown originale',
        ),
      ],
      selected: {_readable},
      onSelectionChanged: (v) => setState(() {
        _readable = v.first;
        if (_readable) _showFind = false;
      }),
    );
  }

  /// Nella vista formattata chi può modificare vede come farlo.
  Widget _formattedHint(ThemeData theme) {
    return Material(
      color: theme.colorScheme.secondaryContainer,
      child: Padding(
        padding: const EdgeInsets.symmetric(horizontal: 12, vertical: 6),
        child: Row(children: [
          Icon(Icons.edit_outlined,
              size: 18, color: theme.colorScheme.onSecondaryContainer),
          const SizedBox(width: 8),
          Expanded(
            child: Text(
              'Clicca un paragrafo, un titolo o una tabella per modificarlo. '
              'Esc o un clic fuori per chiudere.',
              style: theme.textTheme.bodySmall
                  ?.copyWith(color: theme.colorScheme.onSecondaryContainer),
            ),
          ),
        ]),
      ),
    );
  }

  Widget _findBar(ThemeData theme) {
    final info = _find.text.isEmpty
        ? ''
        : _matches.isEmpty
            ? 'Nessun risultato'
            : '${_current + 1} di ${_matches.length}';
    InputDecoration deco(String hint) => InputDecoration(
          hintText: hint,
          isDense: true,
          border: const OutlineInputBorder(),
          contentPadding:
              const EdgeInsets.symmetric(horizontal: 10, vertical: 10),
        );
    return Container(
      color: theme.colorScheme.surfaceContainerLow,
      padding: const EdgeInsets.fromLTRB(12, 8, 8, 8),
      child: Wrap(
        spacing: 8,
        runSpacing: 8,
        crossAxisAlignment: WrapCrossAlignment.center,
        children: [
          SizedBox(
            width: 220,
            child: TextField(
              controller: _find,
              focusNode: _findFocus,
              decoration: deco('Trova'),
              onChanged: (_) => _computeMatches(),
              onSubmitted: (_) {
                _select(_current + (_currentIsSelected ? 1 : 0),
                    focusText: false);
                _findFocus.requestFocus();
              },
            ),
          ),
          SizedBox(
            width: 70,
            child: Text(info, style: theme.textTheme.bodySmall),
          ),
          IconButton(
            tooltip: 'Precedente',
            icon: const Icon(Icons.keyboard_arrow_up),
            onPressed: _matches.isEmpty ? null : () => _select(_current - 1),
          ),
          IconButton(
            tooltip: 'Successivo',
            icon: const Icon(Icons.keyboard_arrow_down),
            onPressed: _matches.isEmpty
                ? null
                : () => _select(_current + (_currentIsSelected ? 1 : 0)),
          ),
          FilterChip(
            label: const Text('Aa'),
            tooltip: 'Maiuscole/minuscole',
            selected: _matchCase,
            onSelected: (v) {
              _matchCase = v;
              _computeMatches();
            },
          ),
          if (!widget.readOnly) ...[
            SizedBox(
              width: 220,
              child: TextField(
                controller: _replace,
                decoration: deco('Sostituisci con'),
                onSubmitted: (_) => _replaceCurrent(),
              ),
            ),
            OutlinedButton(
              onPressed: _matches.isEmpty ? null : _replaceCurrent,
              child: const Text('Sostituisci'),
            ),
            OutlinedButton(
              onPressed: _matches.isEmpty ? null : _replaceAll,
              child: const Text('Sostituisci tutto'),
            ),
          ],
          IconButton(
            tooltip: 'Chiudi (Esc)',
            icon: const Icon(Icons.close),
            onPressed: () => _toggleFind(false),
          ),
        ],
      ),
    );
  }
}

import 'package:flutter/material.dart';
import 'package:flutter/services.dart';

/// Editor del testo OCR con barra Trova/Sostituisci: gli errori di
/// riconoscimento tendono a ripetersi (stessa parola sbagliata in più punti),
/// quindi "sostituisci tutto" fa risparmiare molto tempo.
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
    if (_showFind && text != _lastText) {
      _lastText = text;
      _computeMatches(keepPosition: true);
    }
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
    setState(() => _showFind = show ?? !_showFind);
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
            child: Row(children: [
              TextButton.icon(
                onPressed: () => _toggleFind(),
                icon: const Icon(Icons.find_replace, size: 18),
                label: const Text('Trova e sostituisci'),
              ),
              const Spacer(),
              ListenableBuilder(
                listenable: widget.controller,
                builder: (_, _) {
                  final text = widget.controller.text;
                  final words =
                      RegExp(r'\S+').allMatches(text).length;
                  return Text('$words parole · ${text.length} caratteri',
                      style: theme.textTheme.bodySmall);
                },
              ),
            ]),
          ),
        ),
        if (_showFind) _findBar(theme),
        const Divider(height: 1),
        Expanded(
          child: TextField(
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

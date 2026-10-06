import 'dart:convert';

import 'package:flutter/material.dart';

class _Row {
  _Row(String key, dynamic value, {this.isNew = false})
      : key = TextEditingController(text: key),
        value = TextEditingController(text: _display(value)),
        originalKey = key,
        originalValue = value;

  final TextEditingController key;
  final TextEditingController value;
  final String originalKey;
  final dynamic originalValue;
  final bool isNew;
  bool removed = false;

  static String _display(dynamic v) {
    if (v == null) return '';
    if (v is String) return v;
    return jsonEncode(v);
  }

  /// Ricostruisce il valore conservando il tipo originale: un numero resta
  /// numero, una lista resta lista. Un campo vuoto diventa null solo se lo
  /// era già; i valori nuovi sono sempre testo.
  dynamic get parsedValue {
    final text = value.text.trim();
    final orig = originalValue;
    if (isNew || orig is String) return text;
    if (orig == null) return text.isEmpty ? null : text;
    try {
      return jsonDecode(text);
    } catch (_) {
      return text;
    }
  }

  void dispose() {
    key.dispose();
    value.dispose();
  }
}

/// Editor chiave/valore dei metadati del documento.
class MetadataEditor extends StatefulWidget {
  const MetadataEditor({
    super.key,
    required this.metadata,
    required this.protectedKeys,
    required this.onSave,
    required this.onDirtyChanged,
    this.lockedReason,
  });

  final Map<String, dynamic> metadata;
  final List<String> protectedKeys;
  final Future<bool> Function(Map<String, dynamic> metadata) onSave;
  final ValueChanged<bool> onDirtyChanged;
  final String? lockedReason;

  @override
  State<MetadataEditor> createState() => MetadataEditorState();
}

class MetadataEditorState extends State<MetadataEditor> {
  static final _keyPattern = RegExp(r'^[A-Za-z0-9_][A-Za-z0-9_\-. ]{0,63}$');
  static final _isoDate = RegExp(r'^\d{4}-\d{2}-\d{2}$');

  final _formKey = GlobalKey<FormState>();
  List<_Row> _rows = [];
  bool _saving = false;
  bool _dirty = false;

  @override
  void initState() {
    super.initState();
    _reset();
  }

  @override
  void didUpdateWidget(covariant MetadataEditor oldWidget) {
    super.didUpdateWidget(oldWidget);
    // Un ricaricamento del documento con gli stessi metadati (es. dopo un
    // cambio di stato) non deve cancellare le modifiche in corso.
    if (!_sameMap(oldWidget.metadata, widget.metadata)) {
      // Siamo dentro il build del genitore: la notifica "non più modificato"
      // va rimandata a dopo il frame, altrimenti setState durante il build.
      _reset(notify: false);
      WidgetsBinding.instance.addPostFrameCallback((_) {
        if (mounted) widget.onDirtyChanged(false);
      });
    }
  }

  @override
  void dispose() {
    for (final r in _rows) {
      r.dispose();
    }
    super.dispose();
  }

  void _reset({bool notify = true}) {
    for (final r in _rows) {
      r.dispose();
    }
    final keys = widget.metadata.keys.toList()..sort();
    _rows = [for (final k in keys) _Row(k, widget.metadata[k])];
    if (notify) {
      _setDirty(false);
    } else {
      _dirty = false;
    }
  }

  void _setDirty(bool value) {
    if (_dirty != value) {
      _dirty = value;
      widget.onDirtyChanged(value);
    }
  }

  void _changed() {
    final current = _collect();
    _setDirty(current == null || !_sameMap(current, widget.metadata));
    setState(() {});
  }

  bool _sameMap(Map<String, dynamic> a, Map<String, dynamic> b) =>
      jsonEncode(_sorted(a)) == jsonEncode(_sorted(b));

  Map<String, dynamic> _sorted(Map<String, dynamic> m) =>
      Map.fromEntries(m.entries.toList()..sort((x, y) => x.key.compareTo(y.key)));

  /// Mappa risultante; null se ci sono chiavi duplicate (la validazione
  /// del form mostrerà l'errore).
  Map<String, dynamic>? _collect() {
    final out = <String, dynamic>{};
    for (final r in _rows.where((r) => !r.removed)) {
      final k = r.key.text.trim();
      if (k.isEmpty && r.value.text.trim().isEmpty && r.isNew) continue;
      if (out.containsKey(k)) return null;
      out[k] = r.parsedValue;
    }
    return out;
  }

  void _add() {
    setState(() => _rows.add(_Row('', '', isNew: true)));
    _changed();
  }

  Future<void> _save() async {
    if (!_formKey.currentState!.validate()) return;
    final meta = _collect();
    if (meta == null) return;
    setState(() => _saving = true);
    final ok = await widget.onSave(meta);
    if (mounted) setState(() => _saving = false);
    if (ok) _setDirty(false);
  }

  String? _validateKey(_Row row, String? v) {
    final k = (v ?? '').trim();
    if (k.isEmpty) {
      return row.value.text.trim().isEmpty && row.isNew ? null : 'Nome obbligatorio';
    }
    if (!_keyPattern.hasMatch(k)) return 'Solo lettere, cifre, _ - . spazio';
    if (widget.protectedKeys.contains(k) || RegExp(r'^Header \d+$').hasMatch(k)) {
      return 'Chiave di sistema';
    }
    final dup = _rows.where((r) => !r.removed && r.key.text.trim() == k).length;
    if (dup > 1) return 'Nome duplicato';
    return null;
  }

  String? _validateValue(_Row row, String? v) {
    final k = row.key.text.trim();
    final text = (v ?? '').trim();
    if (k.startsWith('data') && text.isNotEmpty && !_isoDate.hasMatch(text)) {
      return 'Formato AAAA-MM-GG';
    }
    return null;
  }

  @override
  Widget build(BuildContext context) {
    final theme = Theme.of(context);
    final locked = widget.lockedReason != null;
    final visible = _rows.where((r) => !r.removed).toList();

    return Column(crossAxisAlignment: CrossAxisAlignment.stretch, children: [
      if (locked)
        MaterialBanner(
          content: Text(widget.lockedReason!),
          leading: const Icon(Icons.lock_clock_outlined),
          actions: const [SizedBox.shrink()],
        ),
      Expanded(
        child: Form(
          key: _formKey,
          child: ListView(
            padding: const EdgeInsets.all(16),
            children: [
              Text(
                'I metadati sono usati dai filtri e dalle risposte della chat. '
                'Le date vanno nel formato AAAA-MM-GG (es. 2024-03-15). '
                'Il salvataggio aggiorna subito l\'archivio, senza re-indicizzare il testo.',
                style: theme.textTheme.bodySmall
                    ?.copyWith(color: theme.colorScheme.onSurfaceVariant),
              ),
              const SizedBox(height: 16),
              for (final row in visible)
                Padding(
                  padding: const EdgeInsets.only(bottom: 10),
                  child: Row(crossAxisAlignment: CrossAxisAlignment.start, children: [
                    SizedBox(
                      width: 210,
                      child: TextFormField(
                        controller: row.key,
                        enabled: !locked,
                        readOnly: !row.isNew,
                        decoration: InputDecoration(
                          labelText: row.isNew ? 'Nuovo campo' : null,
                          isDense: true,
                          filled: !row.isNew,
                          border: const OutlineInputBorder(),
                        ),
                        style: const TextStyle(fontWeight: FontWeight.w600),
                        validator: (v) => _validateKey(row, v),
                        onChanged: (_) => _changed(),
                      ),
                    ),
                    const SizedBox(width: 8),
                    Expanded(
                      child: TextFormField(
                        controller: row.value,
                        enabled: !locked,
                        minLines: 1,
                        maxLines: 4,
                        decoration: InputDecoration(
                          isDense: true,
                          border: const OutlineInputBorder(),
                          helperText: row.originalValue is String || row.isNew
                              ? null
                              : row.originalValue == null
                                  ? 'vuoto'
                                  : 'valore ${_typeLabel(row.originalValue)}',
                        ),
                        validator: (v) => _validateValue(row, v),
                        onChanged: (_) => _changed(),
                      ),
                    ),
                    IconButton(
                      tooltip: 'Rimuovi campo',
                      icon: const Icon(Icons.delete_outline),
                      onPressed: locked
                          ? null
                          : () {
                              setState(() => row.removed = true);
                              _changed();
                            },
                    ),
                  ]),
                ),
              Align(
                alignment: Alignment.centerLeft,
                child: TextButton.icon(
                  onPressed: locked ? null : _add,
                  icon: const Icon(Icons.add),
                  label: const Text('Aggiungi campo'),
                ),
              ),
            ],
          ),
        ),
      ),
      const Divider(height: 1),
      Padding(
        padding: const EdgeInsets.all(12),
        child: Row(children: [
          if (_dirty)
            Text('Modifiche non salvate',
                style: TextStyle(color: theme.colorScheme.tertiary)),
          const Spacer(),
          TextButton(
            onPressed: _dirty && !_saving ? () => setState(_reset) : null,
            child: const Text('Annulla modifiche'),
          ),
          const SizedBox(width: 8),
          FilledButton.icon(
            onPressed: _dirty && !_saving && !locked ? _save : null,
            icon: _saving
                ? const SizedBox(
                    width: 16, height: 16,
                    child: CircularProgressIndicator(strokeWidth: 2))
                : const Icon(Icons.save_outlined),
            label: const Text('Salva metadati'),
          ),
        ]),
      ),
    ]);
  }

  String _typeLabel(dynamic v) => switch (v) {
        bool _ => 'sì/no (true/false)',
        num _ => 'numerico',
        List _ => 'lista JSON',
        _ => 'JSON',
      };
}

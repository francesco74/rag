import 'dart:convert';

import 'package:flutter/material.dart';

import '../api.dart';
import 'common.dart';

class HistoryPanel extends StatefulWidget {
  const HistoryPanel({super.key, required this.docKey, required this.refreshToken});

  final DocKey docKey;

  /// Cambia a ogni salvataggio, per ricaricare lo storico.
  final int refreshToken;

  @override
  State<HistoryPanel> createState() => _HistoryPanelState();
}

class _HistoryPanelState extends State<HistoryPanel> {
  late Future<List<HistoryEntry>> _future = ReviewApi.instance.history(widget.docKey);

  @override
  void didUpdateWidget(covariant HistoryPanel oldWidget) {
    super.didUpdateWidget(oldWidget);
    if (oldWidget.refreshToken != widget.refreshToken) {
      _future = ReviewApi.instance.history(widget.docKey);
    }
  }

  @override
  Widget build(BuildContext context) {
    final theme = Theme.of(context);
    return FutureBuilder<List<HistoryEntry>>(
      future: _future,
      builder: (context, snap) {
        if (snap.hasError) {
          return Center(child: Text(snap.error.toString()));
        }
        if (!snap.hasData) {
          return const Center(child: CircularProgressIndicator());
        }
        final items = snap.data!;
        if (items.isEmpty) {
          return Center(
            child: Text('Nessuna modifica registrata',
                style: TextStyle(color: theme.colorScheme.onSurfaceVariant)),
          );
        }
        return ListView.separated(
          padding: const EdgeInsets.symmetric(vertical: 8),
          itemCount: items.length,
          separatorBuilder: (_, _) => const Divider(height: 1),
          itemBuilder: (context, i) {
            final e = items[i];
            return ListTile(
              leading: Icon(switch (e.action) {
                'content' => Icons.text_snippet_outlined,
                'metadata' => Icons.label_outline,
                _ => Icons.flag_outlined,
              }),
              title: Text(_summary(e)),
              subtitle: Text([
                '${formatDateTime(e.createdAt)} · ${e.username}',
                if (e.note != null && e.note!.isNotEmpty) '“${e.note}”',
              ].join('\n')),
              isThreeLine: e.note != null && e.note!.isNotEmpty,
              trailing: e.action == 'status' ? null : const Icon(Icons.chevron_right),
              onTap: e.action == 'status' ? null : () => _openDetail(e),
            );
          },
        );
      },
    );
  }

  String _summary(HistoryEntry e) {
    switch (e.action) {
      case 'content':
        return 'Testo corretto';
      case 'metadata':
        final parts = <String>[];
        List<String> l(String k) =>
            ((e.details[k] as List?) ?? const []).map((x) => x.toString()).toList();
        if (l('added').isNotEmpty) parts.add('aggiunti: ${l('added').join(', ')}');
        final changed = l('changed').where((k) => !l('added').contains(k)).toList();
        if (changed.isNotEmpty) parts.add('modificati: ${changed.join(', ')}');
        if (l('removed').isNotEmpty) parts.add('rimossi: ${l('removed').join(', ')}');
        return 'Metadati${parts.isEmpty ? '' : ' — ${parts.join('; ')}'}';
      default:
        final to = e.details['to']?.toString();
        return to == null ? 'Cambio stato' : 'Stato: ${statusLabel(to)}';
    }
  }

  Future<void> _openDetail(HistoryEntry e) async {
    showDialog<void>(
      context: context,
      builder: (ctx) => Dialog(
        insetPadding: const EdgeInsets.all(24),
        child: SizedBox(
          width: 1100,
          height: 760,
          child: FutureBuilder<HistoryEntry>(
            future: ReviewApi.instance.historyEntry(e.id),
            builder: (ctx, snap) {
              if (snap.hasError) return Center(child: Text(snap.error.toString()));
              if (!snap.hasData) return const Center(child: CircularProgressIndicator());
              final full = snap.data!;
              return Column(children: [
                ListTile(
                  title: Text(_summary(full)),
                  subtitle: Text('${formatDateTime(full.createdAt)} · ${full.username}'),
                  trailing: IconButton(
                    icon: const Icon(Icons.close),
                    onPressed: () => Navigator.pop(ctx),
                  ),
                ),
                const Divider(height: 1),
                Expanded(
                  child: full.action == 'metadata'
                      ? _MetadataDiff(full.oldValue, full.newValue)
                      : _TextDiff(full.oldValue ?? '', full.newValue ?? ''),
                ),
              ]);
            },
          ),
        ),
      ),
    );
  }
}

class _MetadataDiff extends StatelessWidget {
  const _MetadataDiff(this.oldJson, this.newJson);
  final String? oldJson;
  final String? newJson;

  Map<String, dynamic> _parse(String? s) {
    try {
      return Map<String, dynamic>.from(jsonDecode(s ?? '{}') as Map);
    } catch (_) {
      return {};
    }
  }

  @override
  Widget build(BuildContext context) {
    final before = _parse(oldJson);
    final after = _parse(newJson);
    final keys = {...before.keys, ...after.keys}.toList()..sort();
    String show(dynamic v) => v == null ? '—' : (v is String ? v : jsonEncode(v));
    return SingleChildScrollView(
      padding: const EdgeInsets.all(16),
      child: Table(
        columnWidths: const {0: FixedColumnWidth(200)},
        border: TableBorder.all(color: Theme.of(context).dividerColor),
        children: [
          const TableRow(children: [
            _Cell('Campo', bold: true),
            _Cell('Prima', bold: true),
            _Cell('Dopo', bold: true),
          ]),
          for (final k in keys)
            TableRow(
              decoration: BoxDecoration(
                color: !before.containsKey(k)
                    ? const Color(0xFFE6F4EA)
                    : !after.containsKey(k)
                        ? const Color(0xFFFCE8E6)
                        : jsonEncode(before[k]) != jsonEncode(after[k])
                            ? const Color(0xFFFFF4D6)
                            : null,
              ),
              children: [
                _Cell(k, bold: true),
                _Cell(before.containsKey(k) ? show(before[k]) : '(assente)'),
                _Cell(after.containsKey(k) ? show(after[k]) : '(rimosso)'),
              ],
            ),
        ],
      ),
    );
  }
}

class _Cell extends StatelessWidget {
  const _Cell(this.text, {this.bold = false});
  final String text;
  final bool bold;

  @override
  Widget build(BuildContext context) => Padding(
        padding: const EdgeInsets.all(8),
        child: SelectableText(text,
            style: bold ? const TextStyle(fontWeight: FontWeight.w600) : null),
      );
}

enum _Op { same, removed, added }

/// Differenze riga per riga (LCS). Per testi molto lunghi si limita al
/// confronto affiancato, per non bloccare il browser.
class _TextDiff extends StatelessWidget {
  const _TextDiff(this.before, this.after);
  final String before;
  final String after;

  static const _maxCells = 4000000;

  List<(_Op, String)>? _diff() {
    final a = before.split('\n');
    final b = after.split('\n');
    // Si escludono prefisso e suffisso comuni: di solito le correzioni OCR
    // toccano poche righe e la matrice resta piccola.
    var start = 0;
    while (start < a.length && start < b.length && a[start] == b[start]) {
      start++;
    }
    var endA = a.length, endB = b.length;
    while (endA > start && endB > start && a[endA - 1] == b[endB - 1]) {
      endA--;
      endB--;
    }
    final ma = a.sublist(start, endA), mb = b.sublist(start, endB);
    if (ma.length * mb.length > _maxCells) return null;

    final n = ma.length, m = mb.length;
    final lcs = List.generate(n + 1, (_) => List<int>.filled(m + 1, 0));
    for (var i = n - 1; i >= 0; i--) {
      for (var j = m - 1; j >= 0; j--) {
        lcs[i][j] = ma[i] == mb[j]
            ? lcs[i + 1][j + 1] + 1
            : (lcs[i + 1][j] >= lcs[i][j + 1] ? lcs[i + 1][j] : lcs[i][j + 1]);
      }
    }
    final out = <(_Op, String)>[];
    // Contesto: 3 righe prima e dopo la zona modificata.
    for (final l in a.sublist((start - 3).clamp(0, start), start)) {
      out.add((_Op.same, l));
    }
    var i = 0, j = 0;
    while (i < n || j < m) {
      if (i < n && j < m && ma[i] == mb[j]) {
        out.add((_Op.same, ma[i]));
        i++;
        j++;
      } else if (j < m && (i >= n || lcs[i][j + 1] >= lcs[i + 1][j])) {
        out.add((_Op.added, mb[j++]));
      } else {
        out.add((_Op.removed, ma[i++]));
      }
    }
    for (final l in a.sublist(endA, (endA + 3).clamp(endA, a.length))) {
      out.add((_Op.same, l));
    }
    return out;
  }

  @override
  Widget build(BuildContext context) {
    final diff = _diff();
    const mono = TextStyle(fontFamily: 'monospace', fontSize: 13, height: 1.4);
    if (diff == null) {
      Widget pane(String title, String text) => Expanded(
            child: Column(crossAxisAlignment: CrossAxisAlignment.stretch, children: [
              Padding(padding: const EdgeInsets.all(8), child: Text(title)),
              Expanded(
                child: SingleChildScrollView(
                  padding: const EdgeInsets.all(8),
                  child: SelectableText(text, style: mono),
                ),
              ),
            ]),
          );
      return Row(children: [
        pane('Prima', before),
        const VerticalDivider(width: 1),
        pane('Dopo', after),
      ]);
    }
    return ListView.builder(
      padding: const EdgeInsets.symmetric(vertical: 8),
      itemCount: diff.length,
      itemBuilder: (_, k) {
        final (op, line) = diff[k];
        final (color, sign) = switch (op) {
          _Op.added => (const Color(0xFFE6F4EA), '+'),
          _Op.removed => (const Color(0xFFFCE8E6), '−'),
          _Op.same => (null, ' '),
        };
        return Container(
          color: color,
          padding: const EdgeInsets.symmetric(horizontal: 12, vertical: 1),
          child: Text('$sign $line', style: mono),
        );
      },
    );
  }
}

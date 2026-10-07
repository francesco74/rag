/// Suddivisione di un testo Markdown in blocchi modificabili uno alla volta.
///
/// Ogni blocco conserva il testo sorgente esatto e gli spazi/righe vuote che
/// lo precedono: ricomponendo i blocchi si ottiene sempre il testo originale,
/// carattere per carattere. Così modificare un paragrafo nella vista
/// formattata cambia solo quel paragrafo, mai il resto del documento.
library;

enum BlockKind { paragraph, heading, table, code }

class MdBlock {
  MdBlock(this.kind, this.text, this.sepBefore);

  final BlockKind kind;

  /// Sorgente Markdown del blocco, senza a capo finale.
  String text;

  /// Testo tra la fine del blocco precedente e l'inizio di questo
  /// (a capo e righe vuote).
  String sepBefore;
}

class MdDocument {
  MdDocument(this.blocks, this.trailing);

  final List<MdBlock> blocks;

  /// Testo dopo l'ultimo blocco (a capo finali).
  String trailing;

  static final _fence = RegExp(r'^\s{0,3}(```|~~~)');
  static final _heading = RegExp(r'^\s{0,3}#{1,6}(\s|$)');
  static final _tableSeparator =
      RegExp(r'^\s*\|?\s*:?-{3,}:?\s*(\|\s*:?-{3,}:?\s*)*\|?\s*$');

  static bool _isTableLine(String line) => line.trimLeft().startsWith('|');

  String get text {
    final out = StringBuffer();
    for (final b in blocks) {
      out
        ..write(b.sepBefore)
        ..write(b.text);
    }
    out.write(trailing);
    return out.toString();
  }

  /// Blocchi: titoli (una riga), tabelle, blocchi di codice e paragrafi
  /// (righe consecutive non vuote, elenchi compresi).
  static MdDocument parse(String text) {
    final starts = <int>[0];
    for (var i = 0; i < text.length; i++) {
      if (text.codeUnitAt(i) == 0x0A) starts.add(i + 1);
    }
    final lineCount = starts.length;
    int lineEnd(int i) => i + 1 < lineCount ? starts[i + 1] - 1 : text.length;
    String line(int i) => text.substring(starts[i], lineEnd(i));

    final blocks = <MdBlock>[];
    var prevEnd = 0;
    void add(BlockKind kind, int first, int last) {
      final start = starts[first];
      final end = lineEnd(last);
      blocks.add(MdBlock(
          kind, text.substring(start, end), text.substring(prevEnd, start)));
      prevEnd = end;
    }

    var i = 0;
    while (i < lineCount) {
      final l = line(i);
      if (l.trim().isEmpty) {
        i++;
        continue;
      }
      final fence = _fence.firstMatch(l);
      if (fence != null) {
        final marker = fence.group(1)!;
        var j = i + 1;
        while (j < lineCount && !line(j).trimLeft().startsWith(marker)) {
          j++;
        }
        if (j >= lineCount) j = lineCount - 1; // blocco di codice non chiuso
        add(BlockKind.code, i, j);
        i = j + 1;
        continue;
      }
      if (_heading.hasMatch(l)) {
        add(BlockKind.heading, i, i);
        i++;
        continue;
      }
      if (_isTableLine(l)) {
        var j = i;
        while (j + 1 < lineCount && _isTableLine(line(j + 1))) {
          j++;
        }
        final isTable = j > i && _tableSeparator.hasMatch(line(i + 1));
        add(isTable ? BlockKind.table : BlockKind.paragraph, i, j);
        i = j + 1;
        continue;
      }
      var j = i;
      while (j + 1 < lineCount) {
        final next = line(j + 1);
        if (next.trim().isEmpty ||
            _fence.hasMatch(next) ||
            _heading.hasMatch(next) ||
            _isTableLine(next)) {
          break;
        }
        j++;
      }
      add(BlockKind.paragraph, i, j);
      i = j + 1;
    }
    return MdDocument(blocks, text.substring(prevEnd));
  }

  /// Sostituisce il testo di [block] e lo ri-suddivide: un paragrafo in cui
  /// si è inserita una riga vuota diventa due blocchi, uno svuotato sparisce.
  /// Gli altri blocchi restano gli stessi oggetti.
  void replaceBlock(MdBlock block, String newText) {
    final i = blocks.indexOf(block);
    if (i < 0) return;
    final sub = parse(newText);
    blocks.removeAt(i);
    var carry = block.sepBefore;
    if (sub.blocks.isNotEmpty) {
      sub.blocks.first.sepBefore = carry + sub.blocks.first.sepBefore;
      blocks.insertAll(i, sub.blocks);
      carry = '';
    }
    carry += sub.trailing;
    final next = i + sub.blocks.length;
    if (next < blocks.length) {
      blocks[next].sepBefore = carry + blocks[next].sepBefore;
    } else {
      trailing = carry + trailing;
    }
  }
}

/// Tabella Markdown modificabile cella per cella. Le righe non toccate
/// restano identiche al sorgente; la riga di separazione non si modifica.
class MdTable {
  MdTable(String source) : _lines = source.split('\n') {
    for (var r = 0; r < _lines.length; r++) {
      rows.add(r == 1 ? const [] : splitRow(_lines[r]));
    }
  }

  final List<String> _lines;

  /// Celle di ogni riga; la riga 1 (separatore) è vuota.
  final List<List<String>> rows = [];

  static final _unescapedPipe = RegExp(r'(?<!\\)\|');

  static List<String> splitRow(String line) {
    var t = line.trim();
    if (t.startsWith('|')) t = t.substring(1);
    if (t.endsWith('|') && !t.endsWith(r'\|')) t = t.substring(0, t.length - 1);
    return t.split(_unescapedPipe).map((c) => c.trim()).toList();
  }

  int get columnCount =>
      rows.fold(0, (m, r) => r.length > m ? r.length : m);

  void setCell(int row, int col, String value) {
    final cells = rows[row];
    final clean = value.replaceAll('\n', ' ').replaceAll(_unescapedPipe, r'\|');
    if (cells[col] == clean) return;
    cells[col] = clean;
    _lines[row] = '| ${cells.join(' | ')} |';
  }

  String get text => _lines.join('\n');
}

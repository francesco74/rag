import 'dart:math' as math;

import 'package:flutter/material.dart';
import 'package:markdown/markdown.dart' as md;

/// Formattazione di un carattere (maschera di bit).
abstract final class RichStyle {
  static const bold = 1;
  static const italic = 2;
  static const code = 4;

  /// Solo sugli a capo: interruzione di riga "forte" (`  \n` nel sorgente).
  static const hardBreak = 16;

  static const inline = bold | italic | code;
}

/// Carattere che nel testo modificabile rappresenta un elemento non
/// modificabile (commento HTML, `<br>`, immagine): lo stesso usato da
/// Flutter per i WidgetSpan.
const objectChar = '￼';

final _headingLine = RegExp(r'^\s{0,3}(#{1,6})(?:[ \t]+(.*?))?[ \t]*$');
final _quoteLine = RegExp(r'^\s{0,3}>\s?');
final _bulletSource = RegExp(r'^([ \t]*)[-*+][ \t]+');
final _bulletPlain = RegExp(r'^([ \t]*)• ');
final _htmlToken = RegExp(r'<!--[\s\S]*?-->|</?[A-Za-z][^<>\n]*>');
final _trailingBlanks = RegExp(r'[ \t]*$');
final _leadingBlanks = RegExp(r'^[ \t]*');
final _alnum = RegExp(r'[\p{L}\p{N}]', unicode: true);
final _asciiPunct = RegExp(r'''[!-/:-@\[-`{-~]''');

/// Controller per modificare un blocco Markdown come testo formattato.
///
/// Il campo mostra solo il testo, senza simboli: grassetto, corsivo, codice
/// e link sono attributi dei caratteri; i punti elenco sono "• ", i titoli e
/// le citazioni sono proprietà del blocco. Alla modifica il blocco viene
/// riscritto in Markdown con [toMarkdown]; finché non si modifica nulla
/// ([dirty] falso) il sorgente originale resta intatto.
class RichBlockController extends TextEditingController {
  /// Paragrafo, elenco, citazione o titolo.
  RichBlockController.block(String source) : inlineOnly = false {
    var content = source;
    final heading = _headingLine.firstMatch(source);
    if (!source.contains('\n') && heading != null) {
      _headingLevel = heading.group(1)!.length;
      content = heading.group(2) ?? '';
    } else {
      var lines = source.split('\n');
      final nonEmpty = lines.where((l) => l.trim().isNotEmpty);
      if (nonEmpty.isNotEmpty && nonEmpty.every(_quoteLine.hasMatch)) {
        quote = true;
        lines = [for (final l in lines) l.replaceFirst(_quoteLine, '')];
      }
      content = [
        for (final l in lines)
          l.replaceFirstMapped(_bulletSource, (m) => '${m[1]}• '),
      ].join('\n');
    }
    _load(source, content);
  }

  /// Testo su una riga, senza elenchi né titoli (es. una cella di tabella).
  RichBlockController.inline(String source) : inlineOnly = true {
    _load(source, source);
  }

  final bool inlineOnly;
  bool quote = false;
  int _headingLevel = 0;

  /// Vero dopo la prima modifica (testo, formattazione o tipo di blocco).
  bool dirty = false;

  final List<int> _attrs = [];
  final List<int> _links = [];
  final List<String> _hrefs = [];
  final List<String> _tokens = [];

  /// Entità HTML usate nel sorgente: si riscrivono allo stesso modo, così
  /// un blocco con `&amp;` non diventa `&` solo perché lo si è modificato.
  final Map<String, String> _entities = {};

  /// Formattazione scelta con il cursore fermo (Ctrl+B senza selezione):
  /// vale per i caratteri digitati subito dopo.
  int? _pending;
  bool _raw = false;

  /// 0 = paragrafo, 1-6 = titolo.
  int get headingLevel => _headingLevel;
  set headingLevel(int level) {
    if (inlineOnly || level == _headingLevel) return;
    _headingLevel = level;
    dirty = true;
    notifyListeners();
  }

  // --- lettura del Markdown ------------------------------------------------

  void _load(String source, String content) {
    for (final (entity, char) in const [
      ('&amp;', '&'),
      ('&lt;', '<'),
      ('&gt;', '>'),
      ('&quot;', '"'),
    ]) {
      if (source.contains(entity)) _entities[char] = entity;
    }
    final out = StringBuffer();
    final nodes = md.Document(
      extensionSet: md.ExtensionSet.none,
      encodeHtml: false,
    ).parseInline(content.replaceAll(objectChar, ''));

    void add(String text, int attrs, int link) {
      out.write(text);
      for (var i = 0; i < text.length; i++) {
        _attrs.add(text[i] == '\n' ? 0 : attrs);
        _links.add(text[i] == '\n' ? -1 : link);
      }
    }

    void token(String raw) {
      _tokens.add(raw);
      add(objectChar, 0, -1);
    }

    void walk(md.Node node, int attrs, int link) {
      if (node is md.Text) {
        var last = 0;
        for (final m in _htmlToken.allMatches(node.text)) {
          add(node.text.substring(last, m.start), attrs, link);
          token(m[0]!);
          last = m.end;
        }
        add(node.text.substring(last), attrs, link);
        return;
      }
      final e = node as md.Element;
      switch (e.tag) {
        case 'strong':
          for (final c in e.children ?? const <md.Node>[]) {
            walk(c, attrs | RichStyle.bold, link);
          }
        case 'em':
          for (final c in e.children ?? const <md.Node>[]) {
            walk(c, attrs | RichStyle.italic, link);
          }
        case 'code':
          add(e.textContent, attrs | RichStyle.code, link);
        case 'a':
          final href = e.attributes['href'] ?? '';
          final title = e.attributes['title'];
          _hrefs.add(title == null ? href : '$href "$title"');
          final index = _hrefs.length - 1;
          for (final c in e.children ?? const <md.Node>[]) {
            walk(c, attrs, index);
          }
        case 'img':
          token(
            '![${e.attributes['alt'] ?? ''}](${e.attributes['src'] ?? ''})',
          );
        case 'br':
          out.write('\n');
          _attrs.add(RichStyle.hardBreak);
          _links.add(-1);
        default:
          for (final c in e.children ?? const <md.Node>[]) {
            walk(c, attrs, link);
          }
      }
    }

    for (final n in nodes) {
      walk(n, 0, -1);
    }
    _raw = true;
    value = TextEditingValue(
      text: out.toString(),
      selection: TextSelection.collapsed(offset: out.length),
    );
    _raw = false;
  }

  // --- modifiche ----------------------------------------------------------

  @override
  set value(TextEditingValue newValue) {
    if (_raw) {
      super.value = newValue;
      return;
    }
    final old = super.value;
    if (newValue.text != old.text) {
      newValue = _applyEdit(old.text, newValue);
      dirty = true;
    } else if (newValue.selection != old.selection) {
      _pending = null;
    }
    super.value = newValue;
  }

  /// Aggiorna formattazione ed elementi non modificabili seguendo la
  /// modifica: ogni modifica del campo è la sostituzione di un tratto.
  TextEditingValue _applyEdit(String old, TextEditingValue value) {
    var text = value.text;
    final minLen = math.min(old.length, text.length);
    var start = 0;
    while (start < minLen && old.codeUnitAt(start) == text.codeUnitAt(start)) {
      start++;
    }
    var endOld = old.length, endNew = text.length;
    while (endOld > start &&
        endNew > start &&
        old.codeUnitAt(endOld - 1) == text.codeUnitAt(endNew - 1)) {
      endOld--;
      endNew--;
    }
    var inserted = text.substring(start, endNew);
    if (inserted.contains(objectChar)) {
      // Un elemento non modificabile incollato non ha più il suo sorgente.
      inserted = inserted.replaceAll(objectChar, '');
      text = text.replaceRange(start, endNew, inserted);
      value = TextEditingValue(
        text: text,
        selection: TextSelection.collapsed(offset: start + inserted.length),
      );
    }
    if (inlineOnly && inserted.contains('\n')) {
      inserted = inserted.replaceAll('\n', ' ');
      text = text.replaceRange(start, start + inserted.length, inserted);
      value = value.copyWith(text: text);
    }

    final tokensBefore = _countObjects(old, 0, start);
    _tokens.removeRange(
      tokensBefore,
      tokensBefore + _countObjects(old, start, endOld),
    );

    final style =
        (_pending ?? _styleAround(old, start, endOld)) & RichStyle.inline;
    final link =
        start > 0 &&
            endOld < old.length &&
            _links[start - 1] >= 0 &&
            _links[start - 1] == _links[endOld]
        ? _links[start - 1]
        : -1;
    _attrs.replaceRange(start, endOld, [
      for (var i = 0; i < inserted.length; i++) inserted[i] == '\n' ? 0 : style,
    ]);
    _links.replaceRange(start, endOld, [
      for (var i = 0; i < inserted.length; i++) inserted[i] == '\n' ? -1 : link,
    ]);
    return value;
  }

  static int _countObjects(String s, int from, int to) {
    var n = 0;
    for (var i = from; i < to; i++) {
      if (s.codeUnitAt(i) == 0xFFFC) n++;
    }
    return n;
  }

  /// Chi scrive prosegue la formattazione del carattere precedente (come
  /// negli editor di testo); a inizio riga quella del successivo.
  int _styleAround(String text, int start, int end) {
    if (start > 0 && text[start - 1] != '\n' && text[start - 1] != objectChar) {
      return _attrs[start - 1];
    }
    if (end < text.length && text[end] != '\n' && text[end] != objectChar) {
      return _attrs[end];
    }
    return 0;
  }

  bool _styled(int i) => text[i] != '\n' && text[i] != objectChar;

  /// Vero se [flag] è attivo nella selezione (o per la prossima battitura).
  bool isActive(int flag) {
    final s = selection;
    if (!s.isValid) return false;
    if (s.isCollapsed) {
      return ((_pending ?? _styleAround(text, s.start, s.start)) & flag) != 0;
    }
    var any = false;
    for (var i = s.start; i < s.end; i++) {
      if (!_styled(i)) continue;
      any = true;
      if (_attrs[i] & flag == 0) return false;
    }
    return any;
  }

  /// Grassetto/corsivo sulla selezione, o sui prossimi caratteri digitati.
  void toggle(int flag) {
    final s = selection;
    if (!s.isValid) return;
    if (s.isCollapsed) {
      _pending = (_pending ?? _styleAround(text, s.start, s.start)) ^ flag;
      notifyListeners();
      return;
    }
    final on = !isActive(flag);
    for (var i = s.start; i < s.end; i++) {
      if (_styled(i)) {
        _attrs[i] = on ? _attrs[i] | flag : _attrs[i] & ~flag;
      }
    }
    dirty = true;
    notifyListeners();
  }

  bool get isList {
    final lines = _selectedLines();
    return lines.isNotEmpty &&
        lines.every((ls) => _bulletPlain.hasMatch(text.substring(ls)));
  }

  /// Aggiunge o toglie il punto elenco alle righe selezionate.
  void toggleList() {
    if (inlineOnly) return;
    final lines = _selectedLines();
    final remove = isList;
    final edits = <(int, int, String)>[];
    for (final ls in lines) {
      final indent = _leadingBlanks.firstMatch(text.substring(ls))!.end;
      final at = ls + indent;
      if (remove) {
        edits.add((at, at + 2, ''));
      } else if (!_bulletPlain.hasMatch(text.substring(ls))) {
        edits.add((at, at, '• '));
      }
    }
    var t = text;
    for (final (from, to, insert) in edits.reversed) {
      t = t.replaceRange(from, to, insert);
      _attrs.replaceRange(from, to, List.filled(insert.length, 0));
      _links.replaceRange(from, to, List.filled(insert.length, -1));
    }
    final lineEnd = t.indexOf('\n', lines.isEmpty ? 0 : lines.last);
    _raw = true;
    value = TextEditingValue(
      text: t,
      selection: TextSelection.collapsed(
        offset: lineEnd < 0 ? t.length : lineEnd,
      ),
    );
    _raw = false;
    dirty = true;
  }

  /// Inizio di ogni riga toccata dalla selezione.
  List<int> _selectedLines() {
    final s = selection;
    if (!s.isValid) return const [];
    final t = text;
    var ls = s.start == 0 ? 0 : t.lastIndexOf('\n', s.start - 1) + 1;
    final out = [ls];
    while (true) {
      final nl = t.indexOf('\n', ls);
      if (nl < 0 || nl >= s.end) break;
      ls = nl + 1;
      out.add(ls);
    }
    return out;
  }

  // --- scrittura del Markdown ---------------------------------------------

  String toMarkdown() {
    final body = _inlineMarkdown();
    if (inlineOnly) return body;
    var lines = [
      for (final l in body.split('\n'))
        l.replaceFirstMapped(_bulletPlain, (m) => '${m[1]}- '),
    ];
    if (_headingLevel > 0) {
      // In un titolo si va a capo per iniziare un paragrafo.
      final first = '${'#' * _headingLevel} ${lines.first.trim()}';
      final rest = lines.skip(1).join('\n').trim();
      return rest.isEmpty ? first : '$first\n\n$rest';
    }
    if (quote) lines = [for (final l in lines) '> $l'];
    return lines.join('\n');
  }

  String _inlineMarkdown() {
    final p = text;
    var out = '';
    final open = <int>[]; // grassetto/corsivo aperti, nell'ordine
    var openLink = -1;
    var token = 0;

    String marker(int m) => m == RichStyle.bold ? '**' : '*';

    // Chiude i marcatori non più attivi; gli spazi finali restano fuori
    // (`**testo** ` e non `**testo **`, che non sarebbe grassetto).
    void closeExcept(int keep) {
      final idx = open.indexWhere((m) => keep & m == 0);
      if (idx < 0) return;
      final blanks = _trailingBlanks.firstMatch(out)![0]!;
      out = out.substring(0, out.length - blanks.length);
      while (open.length > idx) {
        out += marker(open.removeLast());
      }
      out += blanks;
    }

    void closeAll() {
      closeExcept(0);
      if (openLink >= 0) {
        out += '](${_hrefs[openLink]})';
        openLink = -1;
      }
    }

    var i = 0;
    while (i < p.length) {
      final c = p[i];
      if (c == objectChar) {
        closeAll();
        if (token < _tokens.length) out += _tokens[token++];
        i++;
        continue;
      }
      if (c == '\n') {
        closeAll();
        out += _attrs[i] & RichStyle.hardBreak != 0 ? '  \n' : '\n';
        i++;
        continue;
      }
      final attrs = _attrs[i] & RichStyle.inline;
      final link = _links[i];
      var j = i + 1;
      while (j < p.length &&
          p[j] != '\n' &&
          p[j] != objectChar &&
          (_attrs[j] & RichStyle.inline) == attrs &&
          _links[j] == link) {
        j++;
      }
      if (link != openLink) {
        closeAll();
        if (link >= 0) {
          out += '[';
          openLink = link;
        }
      }
      final seg = p.substring(i, j);
      var from = i;
      if (seg.trim().isNotEmpty) {
        final emphasis = attrs & (RichStyle.bold | RichStyle.italic);
        closeExcept(emphasis);
        final lead = _leadingBlanks.firstMatch(seg)![0]!;
        out += lead;
        from += lead.length;
        for (final m in [RichStyle.bold, RichStyle.italic]) {
          if (emphasis & m != 0 && !open.contains(m)) {
            out += marker(m);
            open.add(m);
          }
        }
      }
      out += attrs & RichStyle.code != 0
          ? _codeSpan(p.substring(from, j))
          : _escape(p, from, j);
      i = j;
    }
    closeAll();
    return out;
  }

  static String _codeSpan(String s) {
    var longest = 0, run = 0;
    for (var i = 0; i < s.length; i++) {
      run = s[i] == '`' ? run + 1 : 0;
      longest = math.max(longest, run);
    }
    final fence = '`' * (longest + 1);
    final pad = s.startsWith('`') || s.endsWith('`') ? ' ' : '';
    return '$fence$pad$s$pad$fence';
  }

  /// Protegge i caratteri che nel Markdown cambierebbero significato.
  String _escape(String p, int from, int to) {
    final out = StringBuffer();
    for (var i = from; i < to; i++) {
      final c = p[i];
      final prev = i > 0 ? p[i - 1] : '';
      final next = i + 1 < p.length ? p[i + 1] : '';
      switch (c) {
        case '*' || '`':
          out.write('\\$c');
        case '_':
          final intraword = _alnum.hasMatch(prev) && _alnum.hasMatch(next);
          out.write(intraword ? c : '\\_');
        case '\\':
          out.write(_asciiPunct.hasMatch(next) ? r'\\' : c);
        case '<' when RegExp(r'[A-Za-z/!?]').hasMatch(next):
          out.write(_entities['<'] ?? r'\<');
        case '#' || '>' when _atLineStart(p, i):
          out.write('\\$c');
        default:
          out.write(_entities[c] ?? c);
      }
    }
    return out.toString();
  }

  static bool _atLineStart(String p, int i) {
    for (var k = i - 1; k >= 0; k--) {
      if (p[k] == '\n') return true;
      if (p[k] != ' ' && p[k] != '\t') return false;
    }
    return true;
  }

  // --- aspetto ------------------------------------------------------------

  @override
  TextSpan buildTextSpan({
    required BuildContext context,
    TextStyle? style,
    required bool withComposing,
  }) {
    final scheme = Theme.of(context).colorScheme;
    final p = text;
    final children = <InlineSpan>[];
    var token = 0;
    var i = 0;
    while (i < p.length) {
      if (p[i] == objectChar) {
        final raw = token < _tokens.length ? _tokens[token] : '';
        token++;
        children.add(
          WidgetSpan(
            alignment: PlaceholderAlignment.middle,
            child: _TokenChip(raw),
          ),
        );
        i++;
        continue;
      }
      final attrs = _attrs[i] & RichStyle.inline;
      final link = _links[i];
      var j = i + 1;
      while (j < p.length &&
          p[j] != objectChar &&
          (_attrs[j] & RichStyle.inline) == attrs &&
          _links[j] == link) {
        j++;
      }
      children.add(
        TextSpan(
          text: p.substring(i, j),
          style: TextStyle(
            fontWeight: attrs & RichStyle.bold != 0 ? FontWeight.bold : null,
            fontStyle: attrs & RichStyle.italic != 0 ? FontStyle.italic : null,
            fontFamily: attrs & RichStyle.code != 0 ? 'monospace' : null,
            backgroundColor: attrs & RichStyle.code != 0
                ? scheme.surfaceContainerHighest
                : null,
            color: link >= 0 ? scheme.primary : null,
            decoration: link >= 0 ? TextDecoration.underline : null,
          ),
        ),
      );
      i = j;
    }
    return TextSpan(style: style, children: children);
  }
}

/// Segnaposto di un elemento non modificabile nel testo.
class _TokenChip extends StatelessWidget {
  const _TokenChip(this.raw);

  final String raw;

  @override
  Widget build(BuildContext context) {
    final theme = Theme.of(context);
    final lower = raw.toLowerCase();
    final label = lower.contains('image') || raw.startsWith('![')
        ? 'immagine'
        : lower.startsWith('<br')
        ? '↵'
        : 'HTML';
    return Tooltip(
      message: raw,
      child: Container(
        padding: const EdgeInsets.symmetric(horizontal: 6, vertical: 1),
        decoration: BoxDecoration(
          color: theme.colorScheme.surfaceContainerHigh,
          borderRadius: BorderRadius.circular(4),
          border: Border.all(color: theme.colorScheme.outlineVariant),
        ),
        child: Text(
          label,
          style: theme.textTheme.labelSmall?.copyWith(
            color: theme.colorScheme.onSurfaceVariant,
          ),
        ),
      ),
    );
  }
}

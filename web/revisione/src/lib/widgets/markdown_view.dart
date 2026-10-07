import 'package:flutter/material.dart';
import 'package:flutter_markdown_plus/flutter_markdown_plus.dart';

// Segnali tipici del Markdown prodotto dal converter: titoli, tabelle,
// elenchi, grassetto/corsivo, link, citazioni, blocchi di codice.
final _heading = RegExp(r'^#{1,6}\s+\S', multiLine: true);
final _tableSeparator =
    RegExp(r'^\s*\|?\s*:?-{3,}:?\s*(\|\s*:?-{3,}:?\s*)+\|?\s*$', multiLine: true);
final _listItem = RegExp(r'^\s*([-*+]|\d{1,3}[.)])\s+\S', multiLine: true);
final _emphasis = RegExp(r'(\*\*|__)\S[^\n]*?\S\1');
final _link = RegExp(r'\[[^\]\n]+\]\([^)\s]+\)');
final _quote = RegExp(r'^>\s', multiLine: true);
final _fence = RegExp(r'^```', multiLine: true);

/// Quanto testo esaminare: basta l'inizio del documento e il controllo
/// resta istantaneo anche su testi molto lunghi.
const _sampleLength = 20000;

/// Vero se il testo sembra Markdown e non testo grezzo dell'OCR.
///
/// Un titolo, una tabella o un blocco di codice bastano; gli altri segnali
/// (elenchi, grassetto, link, citazioni) compaiono anche nel testo grezzo,
/// quindi ne servono almeno due tipi diversi o più ripetizioni.
bool looksLikeMarkdown(String text) {
  final sample =
      text.length > _sampleLength ? text.substring(0, _sampleLength) : text;
  if (_heading.hasMatch(sample) ||
      _tableSeparator.hasMatch(sample) ||
      _fence.allMatches(sample).length >= 2) {
    return true;
  }
  final weak = [_listItem, _emphasis, _link, _quote]
      .map((re) => re.allMatches(sample).length)
      .toList();
  final kinds = weak.where((n) => n > 0).length;
  final total = weak.fold<int>(0, (a, b) => a + b);
  return kinds >= 2 || total >= 5;
}

final _htmlComment = RegExp(r'<!--.*?-->', dotAll: true);
final _htmlBreak = RegExp(r'<br\s*/?>', caseSensitive: false);

/// Ripulisce, solo per la visualizzazione, l'HTML che i convertitori
/// lasciano nel Markdown e che il renderer mostrerebbe come testo: commenti
/// (es. `<!-- image -->`) e `<br>`. Il testo salvato non cambia.
String cleanupMarkdown(String text) {
  final lines = text.replaceAll(_htmlComment, '').split('\n');
  return [
    for (final line in lines)
      // In una riga di tabella un a capo spezzerebbe la tabella.
      line.trimLeft().startsWith('|')
          ? line.replaceAll(_htmlBreak, ' ')
          : line.replaceAll(_htmlBreak, '  \n'),
  ].join('\n');
}

/// Stile del Markdown formattato, coerente con il tema dell'app.
MarkdownStyleSheet markdownStyleSheet(ThemeData theme) {
  final scheme = theme.colorScheme;
  return MarkdownStyleSheet.fromTheme(theme).copyWith(
    p: theme.textTheme.bodyLarge?.copyWith(height: 1.5),
    h1: theme.textTheme.headlineSmall,
    h2: theme.textTheme.titleLarge,
    h3: theme.textTheme.titleMedium,
    h4: theme.textTheme.titleSmall,
    blockSpacing: 12,
    tableBorder: TableBorder.all(color: scheme.outlineVariant),
    tableHeadAlign: TextAlign.left,
    tableCellsPadding: const EdgeInsets.symmetric(horizontal: 8, vertical: 6),
    tableCellsDecoration: BoxDecoration(color: scheme.surface),
    blockquoteDecoration: BoxDecoration(
      color: scheme.surfaceContainerLow,
      border: Border(left: BorderSide(color: scheme.outline, width: 3)),
    ),
    codeblockDecoration: BoxDecoration(
      color: scheme.surfaceContainerHighest,
      borderRadius: BorderRadius.circular(4),
    ),
  );
}

/// I link e le immagini puntano a risorse non raggiungibili dal browser:
/// le immagini diventano un segnaposto, i link non fanno nulla.
Widget markdownImagePlaceholder(Uri uri, String? title, String? alt) => Chip(
      avatar: const Icon(Icons.image_outlined, size: 18),
      label: Text(alt?.isNotEmpty == true ? alt! : 'Immagine'),
    );

void ignoreMarkdownLink(String text, String? href, String title) {}

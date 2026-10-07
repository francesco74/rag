import 'package:flutter/material.dart';
import 'package:flutter/services.dart';
import 'package:flutter_test/flutter_test.dart';
import 'package:revisione/widgets/formatted_editor.dart';
import 'package:revisione/widgets/markdown_blocks.dart';
import 'package:revisione/widgets/markdown_view.dart';
import 'package:revisione/widgets/text_editor.dart';

const markdown = '''
## DETERMINAZIONE DIRIGENZIALE N. 12

**Oggetto:** affidamento lavori di manutenzione della strada provinciale.

<!-- image -->

| Voce | Importo |
|------|--------:|
| Lavori | 10.000,00 |
| IVA<br>22% | 2.200,00 |

- impegnare la spesa
- trasmettere l'atto
''';

const ocrRaw = '''
PROVINCIA DI ESEMPIO
DETERMINAZIONE DIRIGENZIALE N. 12 del 01/03/2024
OGGETTO: affidamento lavori di manutenzione della strada provinciale.
IL DIRIGENTE
- visto il D.Lgs. 267/2000;
premesso che occorre provvedere ai lavori * urgenti *
DETERMINA
1. di impegnare la spesa di euro 12.200,00
''';

Widget _editor(String text) => _editorWith(TextEditingController(text: text));

Widget _editorWith(TextEditingController controller) => MaterialApp(
      home: Scaffold(body: OcrTextEditor(controller: controller)),
    );

void main() {
  group('looksLikeMarkdown', () {
    test('riconosce titoli e tabelle', () {
      expect(looksLikeMarkdown(markdown), isTrue);
      expect(looksLikeMarkdown('| a | b |\n|---|---|\n| 1 | 2 |'), isTrue);
      expect(looksLikeMarkdown('# Titolo\ntesto'), isTrue);
    });

    test('non scambia il testo OCR grezzo per Markdown', () {
      expect(looksLikeMarkdown(ocrRaw), isFalse);
      expect(looksLikeMarkdown(''), isFalse);
      expect(looksLikeMarkdown('#1 del registro\nart. 5 - comma 2'), isFalse);
    });

    test('segnali deboli: servono più tipi diversi', () {
      expect(looksLikeMarkdown('- una voce\naltro testo'), isFalse);
      expect(looksLikeMarkdown('- una voce\n**importante** da sapere'), isTrue);
    });
  });

  group('MdDocument', () {
    const samples = [
      markdown,
      ocrRaw,
      '',
      '\n\n',
      '# Titolo',
      '\n\n# Titolo\n\n\n',
      'testo\r\nsu più righe\r\n\r\n## Titolo\r\n',
      '```\ncodice non chiuso\n\n| a |',
      'paragrafo\n| a | b |\n|---|---|\n| 1 | 2 |\ndopo',
      '- uno\n- due\n\n      continuazione rientrata\n',
    ];

    test('ricomporre i blocchi restituisce il testo identico', () {
      for (final t in samples) {
        expect(MdDocument.parse(t).text, t, reason: t);
      }
    });

    test('tipi di blocco', () {
      final kinds = MdDocument.parse(markdown).blocks.map((b) => b.kind);
      expect(kinds, [
        BlockKind.heading,
        BlockKind.paragraph,
        BlockKind.paragraph, // <!-- image -->
        BlockKind.table,
        BlockKind.paragraph, // elenco
      ]);
    });

    test('una riga vuota inserita divide il blocco, il resto non cambia', () {
      final doc = MdDocument.parse('a\n\nb c\n\nd\n');
      final others = [doc.blocks[0], doc.blocks[2]];
      doc.replaceBlock(doc.blocks[1], 'b\n\nc');
      expect(doc.text, 'a\n\nb\n\nc\n\nd\n');
      expect(doc.blocks.length, 4);
      expect(identical(doc.blocks.first, others[0]), isTrue);
      expect(identical(doc.blocks.last, others[1]), isTrue);
    });

    test('un blocco svuotato sparisce senza perdere il resto', () {
      final doc = MdDocument.parse('a\n\nb\n\nc');
      doc.replaceBlock(doc.blocks[1], '');
      expect(doc.blocks.map((b) => b.text), ['a', 'c']);
      expect(doc.text, 'a\n\n\n\nc');
    });
  });

  group('MdTable', () {
    const table = '| Voce | Importo |\n|------|--------:|\n| Lavori | 10.000,00 |\n|IVA|2.200,00|';

    test('modificare una cella riscrive solo la sua riga', () {
      final t = MdTable(table);
      expect(t.rows[2], ['Lavori', '10.000,00']);
      t.setCell(2, 1, '11.000,00');
      expect(t.text,
          '| Voce | Importo |\n|------|--------:|\n| Lavori | 11.000,00 |\n|IVA|2.200,00|');
    });

    test('il carattere | in una cella viene protetto', () {
      final t = MdTable(table)..setCell(3, 0, 'IVA | 22%');
      expect(t.text.split('\n').last, r'| IVA \| 22% | 2.200,00 |');
      expect(MdTable(t.text).rows[3], [r'IVA \| 22%', '2.200,00']);
    });
  });

  testWidgets('Vista formattata: si apre formattata, sorgente su richiesta',
      (tester) async {
    await tester.pumpWidget(_editor(markdown));

    // Niente simboli Markdown né HTML visibili, nessun campo aperto.
    expect(find.text('Formattato'), findsOneWidget);
    expect(find.textContaining('Clicca un paragrafo'), findsOneWidget);
    expect(find.byType(TextField), findsNothing);
    expect(find.textContaining('##'), findsNothing);
    expect(find.textContaining('<!--'), findsNothing);
    expect(find.textContaining('<br>'), findsNothing);
    expect(find.byType(Table), findsOneWidget);

    await tester.tap(find.text('Sorgente'));
    await tester.pump();
    expect(find.byType(TextField), findsOneWidget);
    expect(find.byType(Table), findsNothing);

    // Trova e sostituisci passa sempre al sorgente.
    await tester.tap(find.text('Formattato'));
    await tester.pump();
    await tester.tap(find.text('Trova e sostituisci'));
    await tester.pump();
    expect(find.byType(Table), findsNothing);
  });

  testWidgets('Modifica di un paragrafo nella vista formattata',
      (tester) async {
    final controller = TextEditingController(text: markdown);
    await tester.pumpWidget(_editorWith(controller));

    await tester.tap(find.textContaining('affidamento lavori', findRichText: true));
    await tester.pump();
    final field = find.byType(TextField);
    expect(field, findsOneWidget);
    // Il campo mostra il testo formattato, senza simboli Markdown.
    expect(tester.widget<TextField>(field).controller!.text,
        'Oggetto: affidamento lavori di manutenzione della strada provinciale.');
    expect(find.byTooltip('Grassetto (Ctrl+B)'), findsOneWidget);

    await tester.enterText(field,
        'Oggetto: affidamento lavori di manutenzione della strada comunale.');
    await tester.pump();
    expect(controller.text,
        markdown.replaceFirst('strada provinciale', 'strada comunale'));

    // Esc chiude e mostra il paragrafo formattato aggiornato.
    await tester.sendKeyEvent(LogicalKeyboardKey.escape);
    await tester.pump();
    await tester.pump();
    expect(find.byType(TextField), findsNothing);
    expect(find.textContaining('strada comunale', findRichText: true), findsOneWidget);
    expect(controller.text,
        markdown.replaceFirst('strada provinciale', 'strada comunale'));
  });

  testWidgets('Aprire e chiudere un blocco senza modifiche non cambia il testo',
      (tester) async {
    final controller = TextEditingController(text: markdown);
    await tester.pumpWidget(_editorWith(controller));
    await tester.tap(find.textContaining('DETERMINAZIONE', findRichText: true));
    await tester.pump();
    expect(find.byType(TextField), findsOneWidget);
    await tester.sendKeyEvent(LogicalKeyboardKey.escape);
    await tester.pump();
    await tester.pump();
    expect(controller.text, markdown);
  });

  testWidgets('Modifica di una cella di tabella', (tester) async {
    final controller = TextEditingController(text: markdown);
    await tester.pumpWidget(_editorWith(controller));

    await tester.tap(find.textContaining('10.000,00', findRichText: true));
    await tester.pump();
    final cell = find.byWidgetPredicate(
        (w) => w is TextField && w.controller?.text == '10.000,00');
    expect(cell, findsOneWidget);
    await tester.enterText(cell, '11.000,00');
    await tester.pump();
    expect(controller.text,
        markdown.replaceFirst('| Lavori | 10.000,00 |', '| Lavori | 11.000,00 |'));

    // Un clic fuori dalla tabella la chiude.
    await tester.tapAt(const Offset(5, 590));
    await tester.pump();
    await tester.pump();
    expect(find.byWidgetPredicate((w) => w is TextField), findsNothing);
  });

  testWidgets('Clic su un altro blocco: chiude il primo e apre il secondo',
      (tester) async {
    final controller = TextEditingController(text: markdown);
    await tester.pumpWidget(_editorWith(controller));
    await tester.tap(find.textContaining('affidamento lavori', findRichText: true));
    await tester.pump();
    // Il titolo diventa due blocchi separati da una riga vuota.
    await tester.enterText(find.byType(TextField),
        'Oggetto: affidamento\n\nlavori di manutenzione della strada provinciale.');
    await tester.pump();
    await tester.tap(find.textContaining('DETERMINAZIONE', findRichText: true));
    await tester.pump();
    await tester.pump();
    final field = find.byType(TextField);
    expect(field, findsOneWidget);
    expect(tester.widget<TextField>(field).controller!.text,
        'DETERMINAZIONE DIRIGENZIALE N. 12');
    expect(find.textContaining('lavori di manutenzione', findRichText: true), findsOneWidget);
    expect(controller.text, markdown.replaceFirst(
        'affidamento lavori', 'affidamento\n\nlavori'));
  });

  testWidgets('Grassetto con Ctrl+B sulla parola selezionata', (tester) async {
    final controller = TextEditingController(text: markdown);
    await tester.pumpWidget(_editorWith(controller));
    await tester.tap(find.textContaining('affidamento lavori', findRichText: true));
    await tester.pump();
    final field = tester.widget<TextField>(find.byType(TextField));
    final text = field.controller!.text;
    final start = text.indexOf('lavori');
    field.controller!.selection =
        TextSelection(baseOffset: start, extentOffset: start + 'lavori'.length);
    await tester.pump();
    await tester.sendKeyDownEvent(LogicalKeyboardKey.controlLeft);
    await tester.sendKeyEvent(LogicalKeyboardKey.keyB);
    await tester.sendKeyUpEvent(LogicalKeyboardKey.controlLeft);
    await tester.pump();
    expect(controller.text,
        markdown.replaceFirst('affidamento lavori', 'affidamento **lavori**'));
  });

  testWidgets('Titolo trasformato in paragrafo dalla barra', (tester) async {
    final controller = TextEditingController(text: markdown);
    await tester.pumpWidget(_editorWith(controller));
    await tester.tap(find.textContaining('DETERMINAZIONE', findRichText: true));
    await tester.pump();
    await tester.tap(find.text('T1'));
    await tester.pump();
    expect(controller.text,
        markdown.replaceFirst('## DETERMINAZIONE', '# DETERMINAZIONE'));
    // La barra non chiude il blocco.
    expect(find.byType(TextField), findsOneWidget);
  });

  testWidgets('Elemento HTML mostrato come segnaposto e conservato',
      (tester) async {
    final c = TextEditingController(text: '# T\n\nprima <!-- image --> dopo<br>fine');
    await tester.pumpWidget(MaterialApp(
        home: Scaffold(body: FormattedEditor(controller: c))));
    await tester.tap(find.textContaining('prima', findRichText: true));
    await tester.pump();
    expect(find.text('immagine'), findsOneWidget);
    expect(find.text('↵'), findsOneWidget);
    final field = tester.widget<TextField>(find.byType(TextField));
    await tester.enterText(find.byType(TextField),
        field.controller!.text.replaceFirst('prima', 'primo'));
    await tester.pump();
    expect(c.text, '# T\n\nprimo <!-- image --> dopo<br>fine');
  });

  testWidgets('Sola lettura: niente modifica con il clic', (tester) async {
    await tester.pumpWidget(MaterialApp(
      home: Scaffold(
        body: OcrTextEditor(
            controller: TextEditingController(text: markdown), readOnly: true),
      ),
    ));
    expect(find.text('Formattato'), findsOneWidget);
    expect(find.textContaining('Clicca un paragrafo'), findsNothing);
    await tester.tap(find.textContaining('affidamento lavori', findRichText: true));
    await tester.pump();
    expect(find.byType(TextField), findsNothing);
  });

  testWidgets('Pannello stretto: la barra non trabocca', (tester) async {
    for (final width in [360.0, 550.0, 800.0]) {
      await tester.pumpWidget(MaterialApp(
        home: Scaffold(
          body: Center(
            child: SizedBox(
              width: width,
              child: OcrTextEditor(
                  controller: TextEditingController(text: markdown)),
            ),
          ),
        ),
      ));
      expect(tester.takeException(), isNull, reason: 'larghezza $width');
    }
  });

  testWidgets('Testo grezzo: solo editor, nessun selettore', (tester) async {
    await tester.pumpWidget(_editor(ocrRaw));
    expect(find.text('Formattato'), findsNothing);
    expect(find.byType(TextField), findsOneWidget);
  });
}

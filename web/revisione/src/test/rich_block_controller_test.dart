import 'package:flutter/widgets.dart';
import 'package:flutter_test/flutter_test.dart';
import 'package:revisione/widgets/rich_block_controller.dart';

/// Riscrive il blocco come se fosse stato modificato.
String roundTrip(String source, {bool inline = false}) {
  final c = inline
      ? RichBlockController.inline(source)
      : RichBlockController.block(source);
  return c.toMarkdown();
}

void main() {
  group('testo mostrato senza simboli', () {
    test('grassetto, corsivo, codice, link', () {
      final c = RichBlockController.block(
          '**Oggetto:** *urgente* `art. 5` [sito](http://x.it) &amp; altro');
      expect(c.text, 'Oggetto: urgente art. 5 sito & altro');
    });

    test('elenchi, citazioni, titoli', () {
      expect(RichBlockController.block('- uno\n  * due').text, '• uno\n  • due');
      expect(RichBlockController.block('> citato\n> ancora').text, 'citato\nancora');
      final h = RichBlockController.block('## Titolo **forte**');
      expect(h.text, 'Titolo forte');
      expect(h.headingLevel, 2);
    });

    test('HTML e immagini diventano elementi non modificabili', () {
      final c = RichBlockController.block('a<br>b <!-- image --> ![alt](x.png)');
      expect(c.text, 'a${objectChar}b $objectChar $objectChar');
    });
  });

  group('riscrittura in Markdown senza perdite', () {
    const stable = [
      '**Oggetto:** affidamento lavori',
      '*corsivo* e **grassetto** e ***entrambi***',
      'testo `codice` testo',
      '[link](http://x.it) e [con titolo](http://y.it "T")',
      '- uno\n- **due**\n  - tre',
      '1. primo\n2. secondo',
      '> citazione\n> su due righe',
      '## Titolo',
      '# Titolo **forte**',
      'a<br>b <!-- image -->',
      '![alt](img.png)',
      'riga  \nspezzata',
      r'prezzo 5 \* 3',
      'snake_case resta così',
      'Tom &amp; Jerry',
      r'C:\percorso',
      '**Oggetto:** testo\n\naltro paragrafo',
    ];
    for (final s in stable) {
      test(s, () => expect(roundTrip(s), s));
    }

    test('celle di tabella', () {
      expect(roundTrip('**Totale**', inline: true), '**Totale**');
      expect(roundTrip(r'IVA \| 22%', inline: true), 'IVA | 22%');
    });
  });

  group('modifiche', () {
    test('scrivere dopo un grassetto lo prosegue, la riscrittura è minima', () {
      final c = RichBlockController.block('**Oggetto:** testo');
      expect(c.dirty, isFalse);
      c.text = 'Oggetto: testo modificato';
      expect(c.dirty, isTrue);
      expect(c.toMarkdown(), '**Oggetto:** testo modificato');
    });

    test('caratteri speciali digitati vengono protetti', () {
      final c = RichBlockController.block('testo');
      c.text = '* non elenco, 5*3, # non titolo, <br> scritto';
      expect(c.toMarkdown(),
          r'\* non elenco, 5\*3, # non titolo, \<br> scritto');
      final h = RichBlockController.block('x');
      h.text = '# non titolo';
      expect(h.toMarkdown(), r'\# non titolo');
    });

    test('grassetto sulla selezione', () {
      final c = RichBlockController.block('affidamento lavori urgenti');
      c.selection = const TextSelection(baseOffset: 12, extentOffset: 18);
      expect(c.isActive(RichStyle.bold), isFalse);
      c.toggle(RichStyle.bold);
      expect(c.isActive(RichStyle.bold), isTrue);
      expect(c.toMarkdown(), 'affidamento **lavori** urgenti');
      c.toggle(RichStyle.bold);
      expect(c.toMarkdown(), 'affidamento lavori urgenti');
    });

    test('grassetto che include spazi: gli spazi restano fuori', () {
      final c = RichBlockController.block('uno due tre');
      c.selection = const TextSelection(baseOffset: 3, extentOffset: 8);
      c.toggle(RichStyle.bold);
      expect(c.toMarkdown(), 'uno **due** tre');
    });

    test('grassetto per i prossimi caratteri (cursore fermo)', () {
      final c = RichBlockController.block('Totale: ');
      c.selection = const TextSelection.collapsed(offset: 8);
      c.toggle(RichStyle.bold);
      c.value = const TextEditingValue(
          text: 'Totale: 100', selection: TextSelection.collapsed(offset: 11));
      expect(c.toMarkdown(), 'Totale: **100**');
    });

    test('elenco puntato sulle righe selezionate', () {
      final c = RichBlockController.block('uno\n**due**\ntre');
      c.selection = const TextSelection(baseOffset: 0, extentOffset: 6);
      c.toggleList();
      expect(c.toMarkdown(), '- uno\n- **due**\ntre');
      c.selection = const TextSelection(baseOffset: 0, extentOffset: 8);
      c.toggleList();
      expect(c.toMarkdown(), 'uno\n**due**\ntre');
    });

    test('cancellare un elemento non modificabile lo toglie dal sorgente', () {
      final c = RichBlockController.block('a <!-- image --> b <br> c');
      c.text = c.text.replaceFirst(objectChar, '');
      expect(c.toMarkdown(), 'a  b <br> c');
    });

    test('da titolo a paragrafo e viceversa', () {
      final c = RichBlockController.block('## Titolo');
      c.headingLevel = 0;
      expect(c.toMarkdown(), 'Titolo');
      c.headingLevel = 1;
      expect(c.toMarkdown(), '# Titolo');
    });

    test('a capo in un titolo: il resto diventa un paragrafo', () {
      final c = RichBlockController.block('## Titolo');
      c.text = 'Titolo\ntesto';
      expect(c.toMarkdown(), '## Titolo\n\ntesto');
    });

    test('nelle celle niente a capo', () {
      final c = RichBlockController.inline('cella');
      c.text = 'cel\nla';
      expect(c.text, 'cel la');
    });
  });
}

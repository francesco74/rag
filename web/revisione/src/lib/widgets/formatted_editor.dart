import 'package:flutter/material.dart';
import 'package:flutter/services.dart';
import 'package:flutter_markdown_plus/flutter_markdown_plus.dart';

import 'markdown_blocks.dart';
import 'markdown_view.dart';
import 'rich_block_controller.dart';

/// Vista formattata del Markdown, modificabile un blocco alla volta.
///
/// Il documento è mostrato formattato; con un clic un paragrafo, un titolo
/// o un elenco si apre per la modifica come testo formattato, senza simboli
/// Markdown (grassetto, corsivo, elenchi e titoli dalla barra o con
/// Ctrl+B / Ctrl+I), e una tabella diventa una griglia di celle. Solo il
/// blocco modificato viene riscritto in Markdown: il resto del documento
/// resta identico a com'era, niente riformattazioni involontarie nel testo
/// salvato, nella re-indicizzazione e nello storico.
class FormattedEditor extends StatefulWidget {
  const FormattedEditor({
    super.key,
    required this.controller,
    this.readOnly = false,
  });

  /// Testo completo del documento, condiviso con l'editor del sorgente.
  final TextEditingController controller;
  final bool readOnly;

  @override
  State<FormattedEditor> createState() => _FormattedEditorState();
}

class _FormattedEditorState extends State<FormattedEditor> {
  late MdDocument _doc;
  late String _joined;

  MdBlock? _editing;
  TextEditingController? _blockText;
  MdTable? _table;
  List<List<RichBlockController>> _cells = const [];

  /// Cella su cui agiscono la barra e Ctrl+B / Ctrl+I.
  RichBlockController? _activeCell;

  @override
  void initState() {
    super.initState();
    _parse();
    widget.controller.addListener(_onDocumentChanged);
  }

  @override
  void didUpdateWidget(covariant FormattedEditor oldWidget) {
    super.didUpdateWidget(oldWidget);
    if (oldWidget.controller != widget.controller) {
      oldWidget.controller.removeListener(_onDocumentChanged);
      widget.controller.addListener(_onDocumentChanged);
      _closeEditor();
      _parse();
    }
  }

  @override
  void dispose() {
    widget.controller.removeListener(_onDocumentChanged);
    _disposeEditors();
    super.dispose();
  }

  void _parse() {
    _joined = widget.controller.text;
    _doc = MdDocument.parse(_joined);
  }

  /// Il testo è cambiato fuori da qui (ricarica, "annulla modifiche",
  /// salvataggio): si ricostruiscono i blocchi.
  void _onDocumentChanged() {
    if (widget.controller.text == _joined) return;
    setState(() {
      _closeEditor();
      _parse();
    });
  }

  /// Riporta nel documento completo le modifiche del blocco aperto.
  void _commit() {
    _joined = _doc.text;
    widget.controller.text = _joined;
  }

  // --- apertura e chiusura del blocco in modifica --------------------------

  void _open(MdBlock block) {
    if (widget.readOnly || identical(block, _editing)) return;
    setState(() {
      _closeEditor();
      _editing = block;
      if (block.kind == BlockKind.table) {
        final table = MdTable(block.text);
        _table = table;
        _cells = [
          for (var r = 0; r < table.rows.length; r++)
            [
              for (var c = 0; c < table.rows[r].length; c++)
                RichBlockController.inline(table.rows[r][c])
                  ..addListener(() => _onCellChanged(r, c)),
            ],
        ];
      } else {
        // Il codice si modifica così com'è; il resto come testo formattato.
        _blockText =
            (block.kind == BlockKind.code
                  ? TextEditingController(text: block.text)
                  : RichBlockController.block(block.text))
              ..addListener(_onBlockChanged);
      }
    });
  }

  void _close() {
    if (_editing == null) return;
    setState(_closeEditor);
  }

  /// Chiude il blocco aperto e lo ri-suddivide (una riga vuota inserita lo
  /// divide in due, un blocco svuotato sparisce). Il testo non cambia.
  void _closeEditor() {
    final block = _editing;
    if (block == null) return;
    _doc.replaceBlock(block, block.text);
    _editing = null;
    // I campi di testo vengono smontati nel prossimo frame: i loro
    // controller si eliminano dopo.
    final old = [?_blockText, for (final row in _cells) ...row];
    _blockText = null;
    _table = null;
    _cells = const [];
    _activeCell = null;
    WidgetsBinding.instance.addPostFrameCallback((_) {
      for (final c in old) {
        c.dispose();
      }
    });
  }

  void _disposeEditors() {
    _blockText?.dispose();
    for (final row in _cells) {
      for (final c in row) {
        c.dispose();
      }
    }
  }

  void _onBlockChanged() {
    final block = _editing;
    final controller = _blockText;
    if (block == null || controller == null) return;
    // Finché non si modifica nulla il sorgente originale resta intatto.
    if (controller is RichBlockController && !controller.dirty) return;
    final source = controller is RichBlockController
        ? controller.toMarkdown()
        : controller.text;
    if (source == block.text) return;
    block.text = source;
    _commit();
  }

  void _onCellChanged(int row, int col) {
    final block = _editing;
    final table = _table;
    final cell = _cells[row][col];
    if (block == null || table == null || !cell.dirty) return;
    table.setCell(row, col, cell.toMarkdown());
    if (table.text == block.text) return;
    block.text = table.text;
    _commit();
  }

  // --- interfaccia ---------------------------------------------------------

  @override
  Widget build(BuildContext context) {
    final theme = Theme.of(context);
    final style = markdownStyleSheet(theme);
    final list = ListView.builder(
      padding: const EdgeInsets.fromLTRB(16, 12, 16, 48),
      itemCount: _doc.blocks.length,
      itemBuilder: (context, i) {
        final block = _doc.blocks[i];
        return Padding(
          padding: const EdgeInsets.symmetric(vertical: 4),
          child: identical(block, _editing)
              ? _editor(block, theme)
              : _rendered(block, style, theme),
        );
      },
    );
    // In sola lettura il testo si seleziona e copia liberamente.
    return widget.readOnly ? SelectionArea(child: list) : list;
  }

  Widget _rendered(MdBlock block, MarkdownStyleSheet style, ThemeData theme) {
    final body = Padding(
      padding: const EdgeInsets.symmetric(horizontal: 8, vertical: 4),
      child: MarkdownBody(
        data: cleanupMarkdown(_dedent(block.text)),
        styleSheet: style,
        onTapLink: ignoreMarkdownLink,
        imageBuilder: markdownImagePlaceholder,
      ),
    );
    if (widget.readOnly) return body;
    return Tooltip(
      message: block.kind == BlockKind.table
          ? 'Clicca per modificare le celle'
          : 'Clicca per modificare',
      waitDuration: const Duration(milliseconds: 800),
      child: InkWell(
        borderRadius: BorderRadius.circular(6),
        mouseCursor: SystemMouseCursors.text,
        hoverColor: theme.colorScheme.primary.withValues(alpha: 0.05),
        onTap: () => _open(block),
        child: body,
      ),
    );
  }

  /// Un blocco staccato dal resto (es. la continuazione rientrata di un
  /// elenco) non deve diventare un blocco di codice per il solo rientro.
  static String _dedent(String text) {
    final lines = text.split('\n');
    final indents = lines
        .where((l) => l.trim().isNotEmpty)
        .map((l) => l.length - l.trimLeft().length);
    final common = indents.isEmpty
        ? 0
        : indents.reduce((a, b) => a < b ? a : b);
    if (common < 4) return text;
    return lines
        .map((l) => l.length >= common ? l.substring(common) : l.trimLeft())
        .join('\n');
  }

  /// Destinatario di grassetto/corsivo: il blocco o la cella attiva.
  RichBlockController? get _target {
    final c = _blockText;
    return c is RichBlockController ? c : _activeCell;
  }

  void _toggle(int flag) => setState(() => _target?.toggle(flag));

  TextStyle? _blockStyle(ThemeData theme, RichBlockController c) {
    final t = theme.textTheme;
    return switch (c.headingLevel) {
      0 => t.bodyLarge?.copyWith(height: 1.5),
      1 => t.headlineSmall,
      2 => t.titleLarge,
      3 => t.titleMedium,
      _ => t.titleSmall,
    };
  }

  Widget _editor(MdBlock block, ThemeData theme) {
    final scheme = theme.colorScheme;
    final controller = _blockText;
    final Widget content;
    if (block.kind == BlockKind.table) {
      content = _tableEditor(theme);
    } else if (controller is RichBlockController) {
      content = ListenableBuilder(
        listenable: controller,
        builder: (context, _) {
          final field = TextField(
            controller: controller,
            autofocus: true,
            maxLines: null,
            keyboardType: TextInputType.multiline,
            style: _blockStyle(theme, controller),
            decoration: const InputDecoration(
              isDense: true,
              border: InputBorder.none,
              contentPadding: EdgeInsets.all(8),
            ),
          );
          if (!controller.quote) return field;
          return Container(
            margin: const EdgeInsets.all(4),
            decoration: BoxDecoration(
              color: scheme.surfaceContainerLow,
              border: Border(left: BorderSide(color: scheme.outline, width: 3)),
            ),
            child: field,
          );
        },
      );
    } else {
      content = TextField(
        controller: controller,
        autofocus: true,
        maxLines: null,
        keyboardType: TextInputType.multiline,
        style: const TextStyle(fontFamily: 'monospace', fontSize: 14),
        decoration: const InputDecoration(
          isDense: true,
          border: InputBorder.none,
          contentPadding: EdgeInsets.all(8),
        ),
      );
    }
    return TapRegion(
      // Un clic fuori dal blocco lo chiude (un clic su un altro blocco
      // chiude questo e apre quello).
      onTapOutside: (_) => _close(),
      child: CallbackShortcuts(
        bindings: {
          const SingleActivator(LogicalKeyboardKey.escape): _close,
          const SingleActivator(LogicalKeyboardKey.keyB, control: true): () =>
              _toggle(RichStyle.bold),
          const SingleActivator(LogicalKeyboardKey.keyB, meta: true): () =>
              _toggle(RichStyle.bold),
          const SingleActivator(LogicalKeyboardKey.keyI, control: true): () =>
              _toggle(RichStyle.italic),
          const SingleActivator(LogicalKeyboardKey.keyI, meta: true): () =>
              _toggle(RichStyle.italic),
        },
        child: Container(
          decoration: BoxDecoration(
            color: scheme.surfaceContainerLowest,
            border: Border.all(color: scheme.primary, width: 1.5),
            borderRadius: BorderRadius.circular(6),
          ),
          child: Column(
            crossAxisAlignment: CrossAxisAlignment.stretch,
            children: [
              if (block.kind != BlockKind.code) _toolbar(theme),
              content,
            ],
          ),
        ),
      ),
    );
  }

  /// Barra di formattazione del blocco aperto. TextFieldTapRegion: un clic
  /// sulla barra non toglie il cursore dal testo.
  Widget _toolbar(ThemeData theme) {
    final text = _blockText is RichBlockController
        ? _blockText as RichBlockController
        : null;
    final listenables = <Listenable>[?text, ?_activeCell];
    return TextFieldTapRegion(
      child: Material(
        color: theme.colorScheme.surfaceContainer,
        borderRadius: const BorderRadius.vertical(top: Radius.circular(5)),
        child: ListenableBuilder(
          listenable: Listenable.merge(listenables),
          builder: (context, _) {
            final target = _target;
            Widget toggle(int flag, IconData icon, String tip) => IconButton(
              tooltip: tip,
              isSelected: target?.isActive(flag) ?? false,
              visualDensity: VisualDensity.compact,
              icon: Icon(icon, size: 20),
              onPressed: target == null ? null : () => _toggle(flag),
            );
            return Wrap(
              crossAxisAlignment: WrapCrossAlignment.center,
              spacing: 4,
              children: [
                if (text != null && !text.quote)
                  SegmentedButton<int>(
                    showSelectedIcon: false,
                    style: const ButtonStyle(
                      visualDensity: VisualDensity(
                        horizontal: -4,
                        vertical: -4,
                      ),
                      tapTargetSize: MaterialTapTargetSize.shrinkWrap,
                    ),
                    segments: const [
                      ButtonSegment(
                        value: 0,
                        label: Text('Testo'),
                        tooltip: 'Paragrafo',
                      ),
                      ButtonSegment(
                        value: 1,
                        label: Text('T1'),
                        tooltip: 'Titolo 1',
                      ),
                      ButtonSegment(
                        value: 2,
                        label: Text('T2'),
                        tooltip: 'Titolo 2',
                      ),
                      ButtonSegment(
                        value: 3,
                        label: Text('T3'),
                        tooltip: 'Titolo 3',
                      ),
                    ],
                    selected: {text.headingLevel.clamp(0, 3)},
                    onSelectionChanged: (v) =>
                        setState(() => text.headingLevel = v.first),
                  ),
                toggle(RichStyle.bold, Icons.format_bold, 'Grassetto (Ctrl+B)'),
                toggle(
                  RichStyle.italic,
                  Icons.format_italic,
                  'Corsivo (Ctrl+I)',
                ),
                if (text != null && text.headingLevel == 0)
                  IconButton(
                    tooltip: 'Elenco puntato',
                    isSelected: text.isList,
                    visualDensity: VisualDensity.compact,
                    icon: const Icon(Icons.format_list_bulleted, size: 20),
                    onPressed: () => setState(text.toggleList),
                  ),
                Padding(
                  padding: const EdgeInsets.symmetric(
                    horizontal: 8,
                    vertical: 10,
                  ),
                  child: Text(
                    'Esc per chiudere',
                    style: theme.textTheme.bodySmall?.copyWith(
                      color: theme.colorScheme.onSurfaceVariant,
                    ),
                  ),
                ),
              ],
            );
          },
        ),
      ),
    );
  }

  Widget _tableEditor(ThemeData theme) {
    final table = _table!;
    final columns = table.columnCount;
    final border = theme.colorScheme.outlineVariant;
    final rows = <TableRow>[];
    for (var r = 0; r < table.rows.length; r++) {
      if (r == 1) continue; // riga di separazione
      final header = r == 0;
      rows.add(
        TableRow(
          decoration: header
              ? BoxDecoration(color: theme.colorScheme.surfaceContainer)
              : null,
          children: [
            for (var c = 0; c < columns; c++)
              c < _cells[r].length
                  ? Focus(
                      onFocusChange: (focused) {
                        if (focused) setState(() => _activeCell = _cells[r][c]);
                      },
                      child: TextField(
                        controller: _cells[r][c],
                        autofocus: r == 0 && c == 0,
                        maxLines: null,
                        style: header
                            ? const TextStyle(fontWeight: FontWeight.w600)
                            : null,
                        decoration: const InputDecoration(
                          isDense: true,
                          border: InputBorder.none,
                          contentPadding: EdgeInsets.symmetric(
                            horizontal: 8,
                            vertical: 8,
                          ),
                        ),
                      ),
                    )
                  : const SizedBox.shrink(),
          ],
        ),
      );
    }
    return Padding(
      padding: const EdgeInsets.all(8),
      child: Table(
        border: TableBorder.all(color: border),
        defaultVerticalAlignment: TableCellVerticalAlignment.top,
        children: rows,
      ),
    );
  }
}

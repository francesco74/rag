import 'package:flutter/material.dart';
import 'package:flutter_quill/flutter_quill.dart';
import 'package:flutter_quill_delta_from_html/flutter_quill_delta_from_html.dart';
import 'package:vsc_quill_delta_to_html/vsc_quill_delta_to_html.dart';

/// Testo dell'editor. Fuori dall'editor vive come HTML, l'unico formato che il
/// backend salva e la chat mostra.
class HtmlEditorController {
  HtmlEditorController(String? initial) : quill = QuillController.basic() {
    html = initial ?? '';
  }

  final QuillController quill;

  String get html {
    if (isEmpty) return '';
    return QuillDeltaToHtmlConverter(quill.document.toDelta().toJson(), ConverterOptions()).convert();
  }

  set html(String value) {
    quill.document =
        value.trim().isEmpty ? Document() : Document.fromDelta(HtmlToDelta().convert(value));
  }

  bool get isEmpty => quill.document.toPlainText().trim().isEmpty;

  void dispose() => quill.dispose();
}

/// Editor visuale con barra degli strumenti: solo ciò che la chat sa mostrare.
class HtmlEditor extends StatelessWidget {
  const HtmlEditor({super.key, required this.controller, this.height = 300});

  final HtmlEditorController controller;
  final double height;

  @override
  Widget build(BuildContext context) {
    final border = Theme.of(context).colorScheme.outline;
    return Container(
      height: height,
      decoration: BoxDecoration(border: Border.all(color: border), borderRadius: BorderRadius.circular(4)),
      child: Column(crossAxisAlignment: CrossAxisAlignment.stretch, children: [
        QuillSimpleToolbar(
          controller: controller.quill,
          config: const QuillSimpleToolbarConfig(
            multiRowsDisplay: true,
            showDividers: true,
            showFontFamily: false,
            showFontSize: false,
            showStrikeThrough: false,
            showInlineCode: false,
            showColorButton: false,
            showBackgroundColorButton: false,
            showAlignmentButtons: false,
            showHeaderStyle: true,
            showListCheck: false,
            showCodeBlock: false,
            showQuote: true,
            showIndent: false,
            showSearchButton: false,
            showSubscript: false,
            showSuperscript: false,
            showClipboardCut: false,
            showClipboardCopy: false,
            showClipboardPaste: false,
          ),
        ),
        const Divider(height: 1),
        Expanded(
          child: Padding(
            padding: const EdgeInsets.all(10),
            child: QuillEditor.basic(controller: controller.quill),
          ),
        ),
      ]),
    );
  }
}

import 'dart:ui_web' as ui_web;

import 'package:flutter/material.dart';
import 'package:web/web.dart' as web;

import '../api.dart';

/// Mostra il file originale in un iframe: il browser visualizza da solo PDF e
/// immagini (zoom, pagine, ricerca) senza librerie aggiuntive. Per gli altri
/// formati offre solo il download.
class OriginalViewer extends StatelessWidget {
  const OriginalViewer({super.key, required this.file});

  final OriginalFile? file;

  static final Set<String> _registered = {};

  String _register(String url) {
    final viewType = 'original-file-${url.hashCode}';
    if (_registered.add(viewType)) {
      ui_web.platformViewRegistry.registerViewFactory(viewType, (int _) {
        final iframe = web.HTMLIFrameElement()
          ..src = url
          ..style.border = 'none'
          ..style.width = '100%'
          ..style.height = '100%';
        return iframe;
      });
    }
    return viewType;
  }

  @override
  Widget build(BuildContext context) {
    final theme = Theme.of(context);
    final f = file;
    if (f == null || f.token == null) {
      return _placeholder(
        theme,
        Icons.hide_source_outlined,
        'File originale non presente in archivio',
        'Il testo resta comunque revisionabile.',
      );
    }

    final header = Material(
      color: theme.colorScheme.surfaceContainer,
      child: Padding(
        padding: const EdgeInsets.symmetric(horizontal: 12, vertical: 4),
        child: Row(children: [
          const Icon(Icons.attach_file, size: 18),
          const SizedBox(width: 6),
          Expanded(
            child: Text('${f.name}  ·  ${_size(f.size)}',
                overflow: TextOverflow.ellipsis,
                style: theme.textTheme.bodySmall),
          ),
          IconButton(
            tooltip: 'Apri in una nuova scheda',
            icon: const Icon(Icons.open_in_new, size: 18),
            onPressed: () => web.window.open(f.url()!, '_blank'),
          ),
          IconButton(
            tooltip: 'Scarica',
            icon: const Icon(Icons.download_outlined, size: 18),
            onPressed: () => web.window.open(f.url(download: true)!, '_blank'),
          ),
        ]),
      ),
    );

    return Column(children: [
      header,
      Expanded(
        child: f.previewable
            ? HtmlElementView(viewType: _register(f.url()!))
            : _placeholder(
                theme,
                Icons.insert_drive_file_outlined,
                'Anteprima non disponibile per ${f.mimeType}',
                'Usa "Scarica" per aprirlo con il programma adatto.',
              ),
      ),
    ]);
  }

  Widget _placeholder(ThemeData theme, IconData icon, String title, String sub) {
    return Container(
      color: theme.colorScheme.surfaceContainerLowest,
      alignment: Alignment.center,
      padding: const EdgeInsets.all(24),
      child: Column(mainAxisSize: MainAxisSize.min, children: [
        Icon(icon, size: 48, color: theme.colorScheme.outline),
        const SizedBox(height: 12),
        Text(title, textAlign: TextAlign.center, style: theme.textTheme.titleSmall),
        const SizedBox(height: 4),
        Text(sub,
            textAlign: TextAlign.center,
            style: theme.textTheme.bodySmall
                ?.copyWith(color: theme.colorScheme.onSurfaceVariant)),
      ]),
    );
  }

  String _size(int bytes) {
    if (bytes < 1024) return '$bytes B';
    if (bytes < 1024 * 1024) return '${(bytes / 1024).toStringAsFixed(0)} KB';
    return '${(bytes / (1024 * 1024)).toStringAsFixed(1)} MB';
  }
}

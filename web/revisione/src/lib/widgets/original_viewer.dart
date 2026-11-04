import 'dart:ui_web' as ui_web;

import 'package:flutter/material.dart';
import 'package:web/web.dart' as web;

import '../api.dart';

/// Mostra il file originale: i PDF in un iframe, con gli strumenti del browser
/// (zoom, pagine, ricerca); le immagini in un visore proprio, perché in un iframe
/// il browser non offre né zoom né adattamento. Per gli altri formati offre solo
/// il download.
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
    if (f == null) {
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
            onPressed: () => web.window.open(f.url(), '_blank'),
          ),
          IconButton(
            tooltip: 'Scarica',
            icon: const Icon(Icons.download_outlined, size: 18),
            onPressed: () => web.window.open(f.url(download: true), '_blank'),
          ),
        ]),
      ),
    );

    return Column(children: [
      header,
      Expanded(
        child: f.previewable
            ? f.mimeType.startsWith('image/')
                ? _ImageViewer(key: ValueKey(f.url()), url: f.url())
                : HtmlElementView(viewType: _register(f.url()))
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

/// Immagine adattata allo spazio: rotella e pulsanti per lo zoom, trascinamento
/// per spostarsi, rotazione per le scansioni storte.
class _ImageViewer extends StatefulWidget {
  const _ImageViewer({super.key, required this.url});

  final String url;

  @override
  State<_ImageViewer> createState() => _ImageViewerState();
}

class _ImageViewerState extends State<_ImageViewer> {
  static const _minScale = 0.5;
  static const _maxScale = 12.0;

  final _transform = TransformationController();
  int _turns = 0;
  Size _viewport = Size.zero;

  @override
  void dispose() {
    _transform.dispose();
    super.dispose();
  }

  /// Zoom attorno al centro della vista.
  void _zoom(double factor) {
    final current = _transform.value.getMaxScaleOnAxis();
    final k = (current * factor).clamp(_minScale, _maxScale) / current;
    final c = _viewport.center(Offset.zero);
    _transform.value = Matrix4.identity()
      ..translate(c.dx, c.dy)
      ..scale(k)
      ..translate(-c.dx, -c.dy)
      ..multiply(_transform.value);
  }

  void _fit() => _transform.value = Matrix4.identity();

  @override
  Widget build(BuildContext context) {
    final theme = Theme.of(context);
    return Stack(children: [
      Positioned.fill(
        child: ColoredBox(
          color: theme.colorScheme.surfaceContainerLowest,
          child: LayoutBuilder(builder: (context, constraints) {
            _viewport = constraints.biggest;
            return InteractiveViewer(
              transformationController: _transform,
              minScale: _minScale,
              maxScale: _maxScale,
              boundaryMargin: const EdgeInsets.all(200),
              child: SizedBox.fromSize(
                size: _viewport,
                child: RotatedBox(
                  quarterTurns: _turns,
                  child: Image.network(
                    widget.url,
                    fit: BoxFit.contain,
                    filterQuality: FilterQuality.medium,
                    loadingBuilder: (_, child, progress) =>
                        progress == null ? child : const Center(child: CircularProgressIndicator()),
                    errorBuilder: (_, _, _) => const Center(
                        child: Text('Immagine non visualizzabile: usa "Scarica" per aprirla.')),
                  ),
                ),
              ),
            );
          }),
        ),
      ),
      Positioned(
        right: 12,
        bottom: 12,
        child: Material(
          elevation: 2,
          borderRadius: BorderRadius.circular(8),
          color: theme.colorScheme.surfaceContainerHigh,
          child: Row(mainAxisSize: MainAxisSize.min, children: [
            IconButton(tooltip: 'Riduci', icon: const Icon(Icons.remove), onPressed: () => _zoom(1 / 1.4)),
            IconButton(tooltip: 'Ingrandisci', icon: const Icon(Icons.add), onPressed: () => _zoom(1.4)),
            IconButton(tooltip: 'Adatta alla pagina', icon: const Icon(Icons.fit_screen_outlined), onPressed: _fit),
            IconButton(
              tooltip: 'Ruota',
              icon: const Icon(Icons.rotate_right),
              onPressed: () => setState(() {
                _turns = (_turns + 1) % 4;
                _fit();
              }),
            ),
          ]),
        ),
      ),
    ]);
  }
}

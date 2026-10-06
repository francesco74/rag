import 'package:flutter/material.dart';

import '../api.dart';

const reviewStatuses = ['da_revisionare', 'in_revisione', 'revisionato'];

String statusLabel(String status) => switch (status) {
      'da_revisionare' => 'Da revisionare',
      'in_revisione' => 'In revisione',
      'revisionato' => 'Revisionato',
      _ => status,
    };

class StatusChip extends StatelessWidget {
  const StatusChip(this.status, {super.key});
  final String status;

  @override
  Widget build(BuildContext context) {
    final (bg, fg, icon) = switch (status) {
      'revisionato' => (
          const Color(0xFFDDF2E3),
          const Color(0xFF1B5E20),
          Icons.check_circle_outline
        ),
      'in_revisione' => (
          const Color(0xFFFFF1D6),
          const Color(0xFF7A4B00),
          Icons.edit_note
        ),
      _ => (
          Theme.of(context).colorScheme.surfaceContainerHighest,
          Theme.of(context).colorScheme.onSurfaceVariant,
          Icons.radio_button_unchecked
        ),
    };
    return Container(
      padding: const EdgeInsets.symmetric(horizontal: 8, vertical: 3),
      decoration:
          BoxDecoration(color: bg, borderRadius: BorderRadius.circular(12)),
      child: Row(mainAxisSize: MainAxisSize.min, children: [
        Icon(icon, size: 14, color: fg),
        const SizedBox(width: 4),
        Text(statusLabel(status),
            style: TextStyle(
                color: fg, fontSize: 12, fontWeight: FontWeight.w600)),
      ]),
    );
  }
}

/// "2026-10-06T14:52:00" -> "06/10/2026 14:52"
String formatDateTime(String? iso) {
  if (iso == null) return '-';
  final dt = DateTime.tryParse(iso);
  if (dt == null) return iso;
  String two(int v) => v.toString().padLeft(2, '0');
  return '${two(dt.day)}/${two(dt.month)}/${dt.year} ${two(dt.hour)}:${two(dt.minute)}';
}

void showError(BuildContext context, Object error) {
  final message = error is ApiException ? error.toString() : 'Errore: $error';
  ScaffoldMessenger.of(context).showSnackBar(SnackBar(
    content: Text(message),
    backgroundColor: Theme.of(context).colorScheme.error,
    duration: const Duration(seconds: 6),
  ));
}

void showInfo(BuildContext context, String message) {
  ScaffoldMessenger.of(context)
      .showSnackBar(SnackBar(content: Text(message)));
}

/// Chiede una nota facoltativa per lo storico. Ritorna null se annullato.
Future<String?> askNote(BuildContext context,
    {required String title, required String confirmLabel, String? message}) {
  final controller = TextEditingController();
  return showDialog<String>(
    context: context,
    builder: (ctx) => AlertDialog(
      title: Text(title),
      content: SizedBox(
        width: 420,
        child: Column(mainAxisSize: MainAxisSize.min, children: [
          if (message != null) ...[Text(message), const SizedBox(height: 12)],
          TextField(
            controller: controller,
            autofocus: true,
            maxLines: 3,
            decoration: const InputDecoration(
              labelText: 'Nota per lo storico (facoltativa)',
              border: OutlineInputBorder(),
            ),
          ),
        ]),
      ),
      actions: [
        TextButton(
            onPressed: () => Navigator.pop(ctx), child: const Text('Annulla')),
        FilledButton(
            onPressed: () => Navigator.pop(ctx, controller.text.trim()),
            child: Text(confirmLabel)),
      ],
    ),
  );
}

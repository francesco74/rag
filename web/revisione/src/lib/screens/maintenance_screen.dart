import 'package:flutter/material.dart';

/// Mostrata all'avvio quando il servizio di revisione non risponde, come la
/// pagina di manutenzione della chat.
class MaintenanceScreen extends StatelessWidget {
  const MaintenanceScreen({super.key, required this.onRetry});

  final VoidCallback onRetry;

  @override
  Widget build(BuildContext context) {
    final theme = Theme.of(context);
    return Scaffold(
      body: Center(
        child: Padding(
          padding: const EdgeInsets.all(32),
          child: Column(mainAxisSize: MainAxisSize.min, children: [
            Icon(Icons.construction, size: 80, color: theme.colorScheme.primary),
            const SizedBox(height: 24),
            Text('Sistema in manutenzione',
                style: theme.textTheme.headlineMedium!
                    .copyWith(fontWeight: FontWeight.bold),
                textAlign: TextAlign.center),
            const SizedBox(height: 16),
            Text(
              'Stiamo effettuando degli aggiornamenti tecnici. '
              'Il servizio tornerà disponibile a breve.',
              style: theme.textTheme.bodyLarge,
              textAlign: TextAlign.center,
            ),
            const SizedBox(height: 32),
            ElevatedButton.icon(
              onPressed: onRetry,
              icon: const Icon(Icons.refresh),
              label: const Text('Riprova'),
            ),
          ]),
        ),
      ),
    );
  }
}

import 'package:flutter/material.dart';

import '../api.dart';
import '../settings.dart';
import '../widgets/user_menu.dart';

/// Mostrata a un utente autenticato che non può consultare alcun documento:
/// nessun ruolo che dia la lettura, oppure nessun archivio assegnato.
class NoAccessScreen extends StatelessWidget {
  const NoAccessScreen({super.key, required this.user});

  final ReviewUser user;

  @override
  Widget build(BuildContext context) {
    final theme = Theme.of(context);
    return Scaffold(
      appBar: AppBar(
        title: Text(AppSettings.projectName),
        actions: [
          const UserMenu(),
          const SizedBox(width: 8),
        ],
      ),
      body: Center(
        child: Padding(
          padding: const EdgeInsets.all(24),
          child: Column(mainAxisSize: MainAxisSize.min, children: [
            Icon(Icons.lock_outline, size: 48, color: theme.colorScheme.outline),
            const SizedBox(height: 16),
            Text('Nessuna funzione disponibile', style: theme.textTheme.titleLarge),
            const SizedBox(height: 8),
            Text(
              user.can(Permission.read)
                  ? 'Ciao ${user.displayName}, non ti è ancora stato assegnato '
                      'alcun archivio su cui lavorare.\n'
                      'Chiedi a un amministratore di abilitarti.'
                  : 'Ciao ${user.displayName}, il tuo profilo non è ancora abilitato '
                      'alla consultazione dei documenti.\n'
                      'Chiedi a un amministratore di assegnarti un ruolo.',
              textAlign: TextAlign.center,
            ),
          ]),
        ),
      ),
    );
  }
}

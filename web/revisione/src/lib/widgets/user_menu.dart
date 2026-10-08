import 'package:flutter/material.dart';

import '../api.dart';

/// Menu dell'utente nella barra in alto: nome, ruoli e uscita.
///
/// [confirmLogout] permette alla schermata di chiedere conferma prima di
/// uscire (per esempio se ci sono modifiche non salvate): se restituisce
/// false il logout viene annullato.
class UserMenu extends StatelessWidget {
  const UserMenu({super.key, this.confirmLogout});

  final Future<bool> Function()? confirmLogout;

  Future<void> _logout(BuildContext context) async {
    if (confirmLogout != null && !await confirmLogout!()) return;
    await ReviewApi.instance.logout();
  }

  @override
  Widget build(BuildContext context) {
    final user = ReviewApi.instance.currentUser.value;
    if (user == null) return const SizedBox.shrink();
    final theme = Theme.of(context);
    return PopupMenuButton<void>(
      tooltip: 'Account',
      position: PopupMenuPosition.under,
      itemBuilder: (_) => [
        PopupMenuItem<void>(
          enabled: false,
          child: Column(
            crossAxisAlignment: CrossAxisAlignment.start,
            children: [
              Text(user.displayName,
                  style: theme.textTheme.titleSmall
                      ?.copyWith(color: theme.colorScheme.onSurface)),
              Text(user.username, style: theme.textTheme.bodySmall),
              const SizedBox(height: 4),
              Text(
                user.roles.isEmpty
                    ? 'Nessun ruolo'
                    : 'Ruoli: ${user.roles.join(', ')}',
                style: theme.textTheme.bodySmall,
              ),
              Text(
                switch (user.topics) {
                  null => 'Archivi: tutti',
                  [] => 'Nessun archivio assegnato',
                  final t => 'Archivi: ${t.join(', ')}',
                },
                style: theme.textTheme.bodySmall,
              ),
            ],
          ),
        ),
        const PopupMenuDivider(),
        PopupMenuItem<void>(
          onTap: () => _logout(context),
          child: const ListTile(
            contentPadding: EdgeInsets.zero,
            leading: Icon(Icons.logout),
            title: Text('Esci'),
          ),
        ),
      ],
      child: Padding(
        padding: const EdgeInsets.symmetric(horizontal: 8),
        child: Row(mainAxisSize: MainAxisSize.min, children: [
          CircleAvatar(
            radius: 14,
            backgroundColor: theme.colorScheme.primaryContainer,
            child: Text(
              user.displayName.isEmpty ? '?' : user.displayName[0].toUpperCase(),
              style: TextStyle(
                  fontSize: 13, color: theme.colorScheme.onPrimaryContainer),
            ),
          ),
          const SizedBox(width: 8),
          Text(user.displayName),
          const Icon(Icons.arrow_drop_down),
        ]),
      ),
    );
  }
}

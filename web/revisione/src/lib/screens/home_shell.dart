import 'package:flutter/material.dart';

import '../api.dart';
import 'admin_screen.dart';
import 'documents_screen.dart';
import 'upload_screen.dart';

/// Le funzioni del sito, alla pari, in una barra laterale. Passando da una
/// all'altra ognuna conserva filtri e file scelti.
class HomeShell extends StatefulWidget {
  const HomeShell({super.key});

  @override
  State<HomeShell> createState() => _HomeShellState();
}

class _HomeShellState extends State<HomeShell> {
  final _api = ReviewApi.instance;
  int _index = 0;
  final Set<int> _opened = {0};

  @override
  Widget build(BuildContext context) {
    final sections = <(String, IconData, Widget Function())>[
      ('Revisione', Icons.fact_check_outlined, () => DocumentsScreen(active: _index == 0)),
      if (_api.can(Permission.upload))
        ('Caricamento', Icons.upload_file, () => UploadScreen(active: _index == 1)),
      if (AdminScreen.isAllowed(_api))
        ('Amministrazione', Icons.admin_panel_settings_outlined, () => const AdminScreen()),
    ];
    return Scaffold(
      body: Row(children: [
        NavigationRail(
          selectedIndex: _index,
          labelType: NavigationRailLabelType.all,
          onDestinationSelected: (i) => setState(() {
            _index = i;
            _opened.add(i);
          }),
          destinations: [
            for (final (label, icon, _) in sections)
              NavigationRailDestination(icon: Icon(icon), label: Text(label)),
          ],
        ),
        const VerticalDivider(width: 1),
        Expanded(
          // Ogni funzione carica i suoi dati solo quando la si apre.
          child: IndexedStack(index: _index, children: [
            for (final (i, section) in sections.indexed)
              _opened.contains(i) ? section.$3() : const SizedBox.shrink(),
          ]),
        ),
      ]),
    );
  }
}

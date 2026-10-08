import 'package:flutter/material.dart';

import 'api.dart';
import 'screens/documents_screen.dart';
import 'screens/login_screen.dart';
import 'screens/no_access_screen.dart';
import 'settings.dart';

void main() {
  runApp(const ReviewApp());
}

class ReviewApp extends StatelessWidget {
  const ReviewApp({super.key});

  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      title: AppSettings.projectName,
      debugShowCheckedModeBanner: false,
      theme: ThemeData(
        colorScheme: ColorScheme.fromSeed(seedColor: const Color(0xFF1F5C99)),
        useMaterial3: true,
        visualDensity: VisualDensity.compact,
      ),
      home: const _SessionGate(),
    );
  }
}

/// Mostra il login, l'elenco documenti o l'avviso di accesso non abilitato
/// a seconda della sessione e dei permessi dell'utente.
class _SessionGate extends StatefulWidget {
  const _SessionGate();

  @override
  State<_SessionGate> createState() => _SessionGateState();
}

class _SessionGateState extends State<_SessionGate> {
  late final Future<bool> _restore = ReviewApi.instance.restoreSession();

  @override
  Widget build(BuildContext context) {
    return FutureBuilder<bool>(
      future: _restore,
      builder: (context, snap) {
        if (snap.connectionState != ConnectionState.done) {
          return const Scaffold(body: Center(child: CircularProgressIndicator()));
        }
        return ValueListenableBuilder<ReviewUser?>(
          valueListenable: ReviewApi.instance.currentUser,
          builder: (context, user, _) {
            if (user == null) {
              // Una sessione scaduta mentre si lavora riporta qui: si chiude
              // qualunque schermata aperta sopra la radice.
              WidgetsBinding.instance.addPostFrameCallback((_) {
                if (mounted) Navigator.of(context).popUntil((r) => r.isFirst);
              });
              return const LoginScreen();
            }
            if (!user.can(Permission.read) || user.hasNoTopics) {
              return NoAccessScreen(user: user);
            }
            return const DocumentsScreen();
          },
        );
      },
    );
  }
}

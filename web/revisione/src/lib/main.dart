import 'package:flutter/material.dart';
import 'package:flutter_localizations/flutter_localizations.dart';
import 'package:flutter_quill/flutter_quill.dart';

import 'api.dart';
import 'screens/home_shell.dart';
import 'screens/login_screen.dart';
import 'screens/maintenance_screen.dart';
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
      locale: const Locale('it'),
      supportedLocales: const [Locale('it')],
      localizationsDelegates: const [
        GlobalMaterialLocalizations.delegate,
        GlobalWidgetsLocalizations.delegate,
        GlobalCupertinoLocalizations.delegate,
        FlutterQuillLocalizations.delegate,
      ],
      theme: ThemeData(
        colorScheme: ColorScheme.fromSeed(seedColor: const Color(0xFF1F5C99)),
        useMaterial3: true,
        visualDensity: VisualDensity.compact,
      ),
      home: const _SessionGate(),
    );
  }
}

/// Verifica che il servizio risponda (altrimenti pagina di manutenzione),
/// poi mostra il login, l'elenco documenti o l'avviso di accesso non
/// abilitato a seconda della sessione e dei permessi dell'utente.
class _SessionGate extends StatefulWidget {
  const _SessionGate();

  @override
  State<_SessionGate> createState() => _SessionGateState();
}

class _SessionGateState extends State<_SessionGate> {
  late Future<bool> _start = _connect();

  /// false = servizio non disponibile. La sessione si ripristina solo a
  /// servizio raggiungibile: altrimenti un errore di rete la cancellerebbe.
  Future<bool> _connect() async {
    if (!await ReviewApi.instance.isAvailable()) return false;
    await ReviewApi.instance.restoreSession();
    return true;
  }

  @override
  Widget build(BuildContext context) {
    return FutureBuilder<bool>(
      future: _start,
      builder: (context, snap) {
        if (snap.connectionState != ConnectionState.done) {
          return const Scaffold(body: Center(child: CircularProgressIndicator()));
        }
        if (snap.data != true) {
          return MaintenanceScreen(
              onRetry: () => setState(() => _start = _connect()));
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
            return const HomeShell();
          },
        );
      },
    );
  }
}

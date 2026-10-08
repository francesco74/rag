import 'package:flutter/material.dart';

import 'app_translations.dart';
import 'chat_auth.dart';

/// Finestra di accesso. [reason] spiega perché viene chiesto (es. un
/// documento riservato). Restituisce true se l'accesso è riuscito.
Future<bool> showLoginDialog(BuildContext context, AppLang lang,
    {String? reason}) async {
  final ok = await showDialog<bool>(
    context: context,
    builder: (_) => _LoginDialog(lang: lang, reason: reason),
  );
  return ok == true;
}

class _LoginDialog extends StatefulWidget {
  const _LoginDialog({required this.lang, this.reason});

  final AppLang lang;
  final String? reason;

  @override
  State<_LoginDialog> createState() => _LoginDialogState();
}

class _LoginDialogState extends State<_LoginDialog> {
  final _username = TextEditingController();
  final _password = TextEditingController();
  bool _busy = false;
  bool _obscure = true;
  String? _error;

  String t(String key) => AppTranslations.get(key, widget.lang);

  @override
  void dispose() {
    _username.dispose();
    _password.dispose();
    super.dispose();
  }

  Future<void> _submit() async {
    if (_busy) return;
    if (_username.text.trim().isEmpty || _password.text.isEmpty) {
      setState(() => _error = t('login_missing'));
      return;
    }
    setState(() {
      _busy = true;
      _error = null;
    });
    try {
      await ChatAuth.instance.login(_username.text.trim(), _password.text);
      if (mounted) Navigator.pop(context, true);
    } on AuthException catch (e) {
      if (mounted) {
        setState(() => _error =
            e.message.isEmpty ? t('login_unreachable') : t('login_failed'));
      }
    } finally {
      if (mounted) setState(() => _busy = false);
    }
  }

  @override
  Widget build(BuildContext context) {
    final theme = Theme.of(context);
    return AlertDialog(
      title: Text(t('login_title')),
      content: ConstrainedBox(
        constraints: const BoxConstraints(maxWidth: 360),
        child: AutofillGroup(
          child: Column(
            mainAxisSize: MainAxisSize.min,
            crossAxisAlignment: CrossAxisAlignment.stretch,
            children: [
              if (widget.reason != null) ...[
                Row(crossAxisAlignment: CrossAxisAlignment.start, children: [
                  Icon(Icons.lock_outline,
                      size: 20, color: theme.colorScheme.primary),
                  const SizedBox(width: 8),
                  Expanded(child: Text(widget.reason!)),
                ]),
                const SizedBox(height: 16),
              ],
              TextField(
                controller: _username,
                autofocus: true,
                autofillHints: const [AutofillHints.username],
                textInputAction: TextInputAction.next,
                decoration: InputDecoration(
                  labelText: t('login_username'),
                  border: const OutlineInputBorder(),
                ),
              ),
              const SizedBox(height: 12),
              TextField(
                controller: _password,
                obscureText: _obscure,
                autofillHints: const [AutofillHints.password],
                onSubmitted: (_) => _submit(),
                decoration: InputDecoration(
                  labelText: t('login_password'),
                  border: const OutlineInputBorder(),
                  suffixIcon: IconButton(
                    icon: Icon(
                        _obscure ? Icons.visibility : Icons.visibility_off),
                    onPressed: () => setState(() => _obscure = !_obscure),
                  ),
                ),
              ),
              if (_error != null) ...[
                const SizedBox(height: 12),
                Text(_error!,
                    style: TextStyle(color: theme.colorScheme.error)),
              ],
            ],
          ),
        ),
      ),
      actions: [
        TextButton(
          onPressed: _busy ? null : () => Navigator.pop(context, false),
          child: Text(t('cancel')),
        ),
        FilledButton(
          onPressed: _busy ? null : _submit,
          child: _busy
              ? const SizedBox(
                  width: 18,
                  height: 18,
                  child: CircularProgressIndicator(strokeWidth: 2))
              : Text(t('login')),
        ),
      ],
    );
  }
}

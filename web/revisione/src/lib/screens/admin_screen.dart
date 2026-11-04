import 'dart:convert';
import 'dart:js_interop';
import 'dart:typed_data';

import 'package:flutter/material.dart';
import 'package:flutter/services.dart';
import 'package:web/web.dart' as web;

import '../api.dart';
import '../widgets/common.dart';
import '../widgets/html_editor.dart';
import '../widgets/user_menu.dart';

typedef _Json = Map<String, dynamic>;

/// Amministrazione: utenti, archivi, siti della chat e credenziali esterne.
/// Le schede dipendono dai permessi, che il backend verifica comunque.
class AdminScreen extends StatelessWidget {
  const AdminScreen({super.key});

  static bool isAllowed(ReviewApi api) =>
      api.can(Permission.manageUsers) || api.can(Permission.manageArchives);

  @override
  Widget build(BuildContext context) {
    final api = ReviewApi.instance;
    final tabs = <(Tab, Widget)>[
      if (api.can(Permission.manageUsers))
        (const Tab(icon: Icon(Icons.people_outline), text: 'Utenti'), const _UsersTab()),
      if (api.can(Permission.manageArchives)) ...[
        (const Tab(icon: Icon(Icons.folder_outlined), text: 'Archivi'), const _TopicsTab()),
        (const Tab(icon: Icon(Icons.language), text: 'Siti'), const _SitesTab()),
        (const Tab(icon: Icon(Icons.key_outlined), text: 'Credenziali'), const _KeysTab()),
      ],
    ];
    return DefaultTabController(
      length: tabs.length,
      child: Scaffold(
        appBar: AppBar(
          title: const Text('Amministrazione'),
          actions: const [UserMenu()],
          bottom: TabBar(tabs: [for (final t in tabs) t.$1]),
        ),
        body: TabBarView(children: [for (final t in tabs) t.$2]),
      ),
    );
  }
}

// ---------------------------------------------------------------- elementi comuni

/// Carica i dati di una scheda e mostra caricamento ed errori sempre allo stesso modo.
abstract class _TabState<T extends StatefulWidget> extends State<T> {
  final api = ReviewApi.instance;
  bool loading = true;
  String? error;

  Future<void> fetch();
  Widget content();

  @override
  void initState() {
    super.initState();
    reload();
  }

  Future<void> reload() async {
    setState(() {
      loading = true;
      error = null;
    });
    try {
      await fetch();
    } catch (e) {
      error = e.toString();
    }
    if (mounted) setState(() => loading = false);
  }

  /// Esegue un'operazione e ricarica; gli errori restano in un avviso.
  Future<bool> run(Future<void> Function() action, {String? done}) async {
    try {
      await action();
      if (mounted && done != null) showInfo(context, done);
      await reload();
      return true;
    } catch (e) {
      if (mounted) showError(context, e);
      return false;
    }
  }

  @override
  Widget build(BuildContext context) {
    if (loading) return const Center(child: CircularProgressIndicator());
    if (error != null) {
      return Center(
        child: Column(mainAxisSize: MainAxisSize.min, children: [
          Text(error!),
          const SizedBox(height: 12),
          OutlinedButton(onPressed: reload, child: const Text('Riprova')),
        ]),
      );
    }
    return content();
  }
}

class _TabPage extends StatelessWidget {
  const _TabPage({this.addLabel, this.onAdd, required this.children});

  final String? addLabel;
  final VoidCallback? onAdd;
  final List<Widget> children;

  @override
  Widget build(BuildContext context) {
    return ListView(padding: const EdgeInsets.all(16), children: [
      if (addLabel != null) ...[
        Align(
          alignment: Alignment.centerLeft,
          child: FilledButton.icon(
              onPressed: onAdd, icon: const Icon(Icons.add), label: Text(addLabel!)),
        ),
        const SizedBox(height: 12),
      ],
      ...children,
    ]);
  }
}

/// Riga di un elenco: un clic in un punto qualsiasi apre il menu delle azioni.
class _MenuRow extends StatelessWidget {
  const _MenuRow({required this.tile, required this.onSelected, required this.items});

  final ListTile tile;
  final ValueChanged<String> onSelected;
  final List<PopupMenuEntry<String>> items;

  @override
  Widget build(BuildContext context) => PopupMenuButton<String>(
        tooltip: '',
        position: PopupMenuPosition.under,
        onSelected: onSelected,
        itemBuilder: (_) => items,
        child: tile,
      );
}

Future<bool> _confirm(BuildContext context, String title, String message, String action) async {
  final ok = await showDialog<bool>(
    context: context,
    builder: (ctx) => AlertDialog(
      title: Text(title),
      content: Text(message),
      actions: [
        TextButton(onPressed: () => Navigator.pop(ctx, false), child: const Text('Annulla')),
        FilledButton(onPressed: () => Navigator.pop(ctx, true), child: Text(action)),
      ],
    ),
  );
  return ok == true;
}

/// Finestra con un modulo. Resta aperta se il salvataggio fallisce, così i dati non si perdono.
Future<void> _formDialog(BuildContext context,
    {required String title,
    required Widget Function(void Function(VoidCallback) setState) builder,
    required Future<bool> Function() onSave}) {
  return showDialog<void>(
    context: context,
    barrierDismissible: false,
    builder: (ctx) => StatefulBuilder(
      builder: (ctx, setState) => AlertDialog(
        title: Text(title),
        content: SizedBox(width: 620, child: SingleChildScrollView(child: builder(setState))),
        actions: [
          TextButton(onPressed: () => Navigator.pop(ctx), child: const Text('Annulla')),
          FilledButton(
            onPressed: () async {
              if (await onSave() && ctx.mounted) Navigator.pop(ctx);
            },
            child: const Text('Salva'),
          ),
        ],
      ),
    ),
  );
}

Widget _field(TextEditingController controller, String label,
    {bool obscure = false, int maxLines = 1, bool enabled = true, String? hint, TextInputType? keyboard}) {
  return Padding(
    padding: const EdgeInsets.only(bottom: 12),
    child: TextField(
      controller: controller,
      obscureText: obscure,
      maxLines: maxLines,
      enabled: enabled,
      keyboardType: keyboard,
      decoration: InputDecoration(labelText: label, helperText: hint, border: const OutlineInputBorder()),
    ),
  );
}

Widget _chips(String label, List<String> options, Set<String> selected, void Function(VoidCallback) setState) {
  return Padding(
    padding: const EdgeInsets.only(bottom: 12),
    child: Column(crossAxisAlignment: CrossAxisAlignment.start, children: [
      Text(label),
      const SizedBox(height: 6),
      if (options.isEmpty) const Text('Nessuno disponibile'),
      Wrap(spacing: 6, runSpacing: 6, children: [
        for (final option in options)
          FilterChip(
            label: Text(option),
            selected: selected.contains(option),
            onSelected: (on) => setState(() => on ? selected.add(option) : selected.remove(option)),
          ),
      ]),
    ]),
  );
}

Future<List<String>> _topicIds(ReviewApi api) async =>
    (await api.topics()).map((t) => t.id).toList()..sort();

// ---------------------------------------------------------------- utenti

class _UsersTab extends StatefulWidget {
  const _UsersTab();

  @override
  State<_UsersTab> createState() => _UsersTabState();
}

class _UsersTabState extends _TabState<_UsersTab> {
  List<_Json> users = [];
  List<String> roles = [];
  List<String> topics = [];

  @override
  Future<void> fetch() async {
    users = List<_Json>.from((await api.adminRequest('GET', '/users', admin: false))['items'] as List);
    roles = List<String>.from((await api.adminRequest('GET', '/roles'))['items'] as List);
    topics = await _topicIds(api);
  }

  Future<void> _edit([_Json? user]) async {
    final username = TextEditingController(text: user?['username'] as String?);
    final name = TextEditingController(text: user?['display_name'] as String?);
    final password = TextEditingController();
    final selectedRoles = Set<String>.from(user?['roles'] as List? ?? const ['revisore']);
    final selectedTopics = Set<String>.from(user?['topics'] as List? ?? const []);
    var active = user?['active'] as bool? ?? true;
    await _formDialog(
      context,
      title: user == null ? 'Nuovo utente' : 'Utente ${user['username']}',
      builder: (setState) => Column(mainAxisSize: MainAxisSize.min, children: [
        _field(username, 'Nome utente', enabled: user == null),
        _field(name, 'Nome visualizzato'),
        if (user == null) _field(password, 'Password', obscure: true, hint: 'Almeno 10 caratteri'),
        _chips('Ruoli', roles, selectedRoles, setState),
        _chips('Archivi (il ruolo admin li vede tutti)', topics, selectedTopics, setState),
        if (user != null)
          SwitchListTile(
            contentPadding: EdgeInsets.zero,
            title: const Text('Attivo'),
            value: active,
            onChanged: (v) => setState(() => active = v),
          ),
      ]),
      onSave: () => run(() async {
        final body = {
          'display_name': name.text,
          'roles': selectedRoles.toList(),
          'topics': selectedTopics.toList(),
          'active': active,
        };
        if (user == null) {
          await api.adminRequest('POST', '/users',
              body: {...body, 'username': username.text, 'password': password.text});
        } else {
          await api.adminRequest('PUT', '/users/${user['id']}', body: body);
        }
      }),
    );
  }

  Future<void> _password(_Json user) async {
    final password = TextEditingController();
    await _formDialog(
      context,
      title: 'Nuova password per ${user['username']}',
      builder: (_) => _field(password, 'Password', obscure: true,
          hint: 'Almeno 10 caratteri. Le sessioni aperte dell\'utente vengono chiuse.'),
      onSave: () => run(
          () => api.adminRequest('POST', '/users/${user['id']}/password', body: {'password': password.text}),
          done: 'Password cambiata'),
    );
  }

  @override
  Widget content() => _TabPage(
        addLabel: 'Nuovo utente',
        onAdd: _edit,
        children: [
          for (final u in users)
            Card(
              child: _MenuRow(
                tile: ListTile(
                  leading: Icon(u['active'] == true ? Icons.person_outline : Icons.person_off_outlined),
                  title: Text('${u['display_name'] ?? u['username']} (${u['username']})'),
                  subtitle: Text('Ruoli: ${(u['roles'] as List).join(', ')}\n'
                      'Archivi: ${u['topics'] == null ? 'tutti' : (u['topics'] as List).join(', ')}\n'
                      'Ultimo accesso: ${formatDateTime(u['last_login_at'] as String?)}'
                      '${u['active'] == true ? '' : '  ·  disattivato'}'),
                  isThreeLine: true,
                ),
                onSelected: (action) => switch (action) {
                  'edit' => _edit(u),
                  'password' => _password(u),
                  _ => run(() => api.adminRequest('POST', '/users/${u['id']}/unlock'), done: 'Accesso sbloccato'),
                },
                items: const [
                  PopupMenuItem(value: 'edit', child: Text('Modifica')),
                  PopupMenuItem(value: 'password', child: Text('Cambia password')),
                  PopupMenuItem(value: 'unlock', child: Text('Sblocca l\'accesso')),
                ],
              ),
            ),
        ],
      );
}

// ---------------------------------------------------------------- archivi

class _TopicsTab extends StatefulWidget {
  const _TopicsTab();

  @override
  State<_TopicsTab> createState() => _TopicsTabState();
}

class _TopicsTabState extends _TabState<_TopicsTab> {
  List<_Json> topics = [];
  List<String> prompts = [];

  @override
  Future<void> fetch() async {
    final res = await api.adminRequest('GET', '/topics');
    topics = List<_Json>.from(res['items'] as List);
    prompts = List<String>.from(res['prompts'] as List);
  }

  Future<void> _editTopic(_Json topic) async {
    final description = TextEditingController(text: topic['description'] as String?);
    final aliases = TextEditingController(text: topic['aliases'] as String?);
    var prompt = topic['prompt'] as String? ?? 'general';
    var isPublic = topic['public_access'] as bool? ?? false;
    await _formDialog(
      context,
      title: 'Archivio ${topic['topic_id']}',
      builder: (setState) => Column(mainAxisSize: MainAxisSize.min, children: [
        _field(description, 'Descrizione', maxLines: 2),
        _field(aliases, 'Sinonimi (facoltativi)'),
        DropdownButtonFormField<String>(
          initialValue: prompts.contains(prompt) ? prompt : null,
          decoration: const InputDecoration(labelText: 'Prompt', border: OutlineInputBorder()),
          items: [for (final p in prompts) DropdownMenuItem(value: p, child: Text(p))],
          onChanged: (v) => setState(() => prompt = v ?? prompt),
        ),
        SwitchListTile(
          contentPadding: EdgeInsets.zero,
          title: const Text('Documenti originali scaricabili da chiunque'),
          value: isPublic,
          onChanged: (v) => setState(() => isPublic = v),
        ),
      ]),
      onSave: () => run(() async {
        final body = {
          'description': description.text,
          'aliases': aliases.text,
          'prompt': prompt,
          'public_access': isPublic,
        };
        await api.adminRequest('PUT', '/topics/${topic['topic_id']}', body: body);
      }),
    );
  }

  Future<void> _editSub(String topicId, [_Json? sub]) async {
    final id = TextEditingController(text: sub?['sub_topic_id'] as String?);
    final description = TextEditingController(text: sub?['description'] as String?);
    final long = TextEditingController(text: sub?['description_long'] as String?);
    final chunk = TextEditingController(text: sub?['chunk_size']?.toString() ?? '');
    final overlap = TextEditingController(text: (sub?['chunk_overlap'] ?? 50).toString());
    final parent = TextEditingController(text: (sub?['parent_chunk_size'] ?? 1500).toString());
    var markdown = sub?['use_markdown_splitter'] as bool? ?? true;
    await _formDialog(
      context,
      title: sub == null ? 'Nuova sottocategoria di $topicId' : 'Sottocategoria ${sub['sub_topic_id']}',
      builder: (setState) => Column(mainAxisSize: MainAxisSize.min, children: [
        _field(id, 'Identificativo', enabled: sub == null,
            hint: 'Lettere minuscole, cifre, _ e -. Non si può cambiare.'),
        _field(description, 'Descrizione'),
        _field(long, 'Descrizione estesa'),
        _field(chunk, 'Dimensione dei chunk (vuoto = predefinita)', keyboard: TextInputType.number),
        _field(overlap, 'Sovrapposizione dei chunk', keyboard: TextInputType.number),
        _field(parent, 'Dimensione dei parent', keyboard: TextInputType.number),
        SwitchListTile(
          contentPadding: EdgeInsets.zero,
          title: const Text('Dividi il testo sui titoli Markdown'),
          value: markdown,
          onChanged: (v) => setState(() => markdown = v),
        ),
      ]),
      onSave: () => run(() async {
        final body = {
          'description': description.text,
          'description_long': long.text,
          'chunk_size': int.tryParse(chunk.text),
          'chunk_overlap': int.tryParse(overlap.text),
          'parent_chunk_size': int.tryParse(parent.text),
          'use_markdown_splitter': markdown,
        };
        if (sub == null) {
          await api.adminRequest('POST', '/topics/$topicId/subtopics', body: {...body, 'sub_topic_id': id.text});
        } else {
          await api.adminRequest('PUT', '/topics/$topicId/subtopics/${sub['sub_topic_id']}', body: body);
        }
      }),
    );
  }

  @override
  Widget content() => _TabPage(
        children: [
          for (final t in topics)
            Card(
              child: ExpansionTile(
                title: Text('${t['topic_id']}'),
                subtitle: Text('${t['description']}\n'
                    'Prompt: ${t['prompt'] ?? 'general'}  ·  '
                    'Originali ${t['public_access'] == true ? 'pubblici' : 'riservati'}'),
                childrenPadding: const EdgeInsets.fromLTRB(16, 0, 16, 12),
                children: [
                  Row(children: [
                    OutlinedButton(onPressed: () => _editTopic(t), child: const Text('Modifica')),
                    const SizedBox(width: 8),
                    OutlinedButton(
                        onPressed: () => _editSub(t['topic_id'] as String),
                        child: const Text('Aggiungi sottocategoria')),
                  ]),
                  for (final s in t['sub_topics'] as List)
                    _MenuRow(
                      tile: ListTile(
                          contentPadding: EdgeInsets.zero,
                          title: Text('${s['sub_topic_id']}'),
                          subtitle: Text('${s['description']}'),
                        ),
                        onSelected: (action) async {
                          if (action == 'edit') {
                            await _editSub(t['topic_id'] as String, s as _Json);
                          } else if (await _confirm(context, 'Eliminare la sottocategoria?',
                              'Si può eliminare solo se non ha documenti.', 'Elimina')) {
                            await run(() => api.adminRequest(
                                'DELETE', '/topics/${t['topic_id']}/subtopics/${s['sub_topic_id']}'));
                          }
                        },
                        items: const [
                          PopupMenuItem(value: 'edit', child: Text('Modifica')),
                          PopupMenuItem(value: 'delete', child: Text('Elimina')),
                        ],
                      ),
                  ],
                ),
              ),
          ],
        );
  }
  
  // ---------------------------------------------------------------- siti
  
  /// Una riga dell'elenco dei pulsanti della home.
class _HomeButton {
  _HomeButton([_Json? button])
      : labelIt = TextEditingController(text: button?['label_it'] as String?),
        labelEn = TextEditingController(text: button?['label_en'] as String?),
        url = TextEditingController(text: button?['url'] as String?);

  final TextEditingController labelIt;
  final TextEditingController labelEn;
  final TextEditingController url;

  _Json toJson() => {'label_it': labelIt.text, 'label_en': labelEn.text, 'url': url.text};
}

class _SitesTab extends StatefulWidget {
    const _SitesTab();
  
    @override
    State<_SitesTab> createState() => _SitesTabState();
  }
  
  class _SitesTabState extends _TabState<_SitesTab> {
    List<_Json> sites = [];
  
    @override
    Future<void> fetch() async {
      sites = List<_Json>.from((await api.adminRequest('GET', '/sites'))['items'] as List);
    }
  
    Future<void> _edit(_Json site) async {
    final name = TextEditingController(text: site['name'] as String?);
    final welcomeIt = HtmlEditorController(site['welcome_it'] as String?);
    final welcomeEn = HtmlEditorController(site['welcome_en'] as String?);
    final buttons = [for (final b in site['buttons'] as List? ?? const []) _HomeButton(b as _Json)];
    var dateFilter = site['allow_date_filter'] as bool? ?? false;
    var subtopicSelection = site['allow_subtopic_selection'] as bool? ?? true;
    var active = site['active'] as bool? ?? true;
    var translating = false;

    Future<void> translate(void Function(VoidCallback) setState) async {
      final hasEnglish = !welcomeEn.isEmpty || buttons.any((b) => b.labelEn.text.trim().isNotEmpty);
      if (hasEnglish &&
          !await _confirm(context, 'Sostituire il testo inglese?',
              'Testo ed etichette inglesi attuali verranno sostituiti dalla traduzione.', 'Sostituisci')) {
        return;
      }
      setState(() => translating = true);
      try {
        final res = await api.adminRequest('POST', '/translate',
            body: {'html': welcomeIt.html, 'labels': [for (final b in buttons) b.labelIt.text]});
        welcomeEn.html = res['html'] as String;
        final labels = res['labels'] as List;
        for (var i = 0; i < buttons.length; i++) {
          buttons[i].labelEn.text = labels[i] as String;
        }
      } catch (e) {
        if (mounted) showError(context, e);
      }
      setState(() => translating = false);
    }

    await _formDialog(
      context,
      title: 'Sito ${site['hostname']}',
      builder: (setState) => Column(mainAxisSize: MainAxisSize.min, crossAxisAlignment: CrossAxisAlignment.start, children: [
        Padding(
          padding: const EdgeInsets.only(bottom: 12),
          child: Text('Archivio: ${site['topic_id']}'),
        ),
        _field(name, 'Nome del sito'),
        const Text('Testo della home in italiano'),
        const SizedBox(height: 4),
        HtmlEditor(controller: welcomeIt),
        const SizedBox(height: 12),
        Row(children: [
          const Expanded(child: Text('Testo della home in inglese (vuoto = quello italiano)')),
          OutlinedButton.icon(
            onPressed: translating ? null : () => translate(setState),
            icon: translating
                ? const SizedBox(width: 16, height: 16, child: CircularProgressIndicator(strokeWidth: 2))
                : const Icon(Icons.translate),
            label: const Text('Traduci in inglese con IA'),
          ),
        ]),
        const SizedBox(height: 4),
        HtmlEditor(controller: welcomeEn),
        const SizedBox(height: 16),
        const Text('Pulsanti sotto le statistiche'),
        const SizedBox(height: 8),
        for (final (index, button) in buttons.indexed)
          Row(crossAxisAlignment: CrossAxisAlignment.start, children: [
            Expanded(flex: 2, child: _field(button.labelIt, 'Etichetta')),
            const SizedBox(width: 8),
            Expanded(flex: 2, child: _field(button.labelEn, 'In inglese')),
            const SizedBox(width: 8),
            Expanded(flex: 3, child: _field(button.url, 'Indirizzo', keyboard: TextInputType.url)),
            IconButton(
              tooltip: 'Sposta in alto',
              icon: const Icon(Icons.arrow_upward),
              onPressed: index == 0
                  ? null
                  : () => setState(() => buttons.insert(index - 1, buttons.removeAt(index))),
            ),
            IconButton(
              tooltip: 'Togli',
              icon: const Icon(Icons.delete_outline),
              onPressed: () => setState(() => buttons.removeAt(index)),
            ),
          ]),
        TextButton.icon(
          onPressed: () => setState(() => buttons.add(_HomeButton())),
          icon: const Icon(Icons.add),
          label: const Text('Aggiungi pulsante'),
        ),
        SwitchListTile(
          contentPadding: EdgeInsets.zero,
          title: const Text('Periodo nei Filtri'),
          value: dateFilter,
          onChanged: (v) => setState(() => dateFilter = v),
        ),
        SwitchListTile(
          contentPadding: EdgeInsets.zero,
          title: const Text('Scelta delle serie nei Filtri'),
          value: subtopicSelection,
          onChanged: (v) => setState(() => subtopicSelection = v),
        ),
        SwitchListTile(
          contentPadding: EdgeInsets.zero,
          title: const Text('Attivo'),
          value: active,
          onChanged: (v) => setState(() => active = v),
        ),
      ]),
      onSave: () => run(() async {
        await api.adminRequest('PUT', '/sites/${site['hostname']}', body: {
          'name': name.text,
          'allow_date_filter': dateFilter,
          'allow_subtopic_selection': subtopicSelection,
          'active': active,
          'welcome_it': welcomeIt.html,
          'welcome_en': welcomeEn.html,
          'buttons': [for (final b in buttons) b.toJson()],
        });
      }),
    );
  }

  /// Il sito sullo stesso schema e porta da cui si usa l'amministrazione.
  void _openSite(_Json site) {
    final base = Uri.base;
    final port = base.hasPort ? ':${base.port}' : '';
    web.window.open('${base.scheme}://${site['hostname']}$port/', '_blank');
  }

  Future<void> _uploadIcon(_Json site) async {
      final bytes = await _pickPng();
      if (bytes == null) return;
      await run(
          () => api.adminRequest('PUT', '/sites/${site['hostname']}/icon', body: {'png': base64Encode(bytes)}),
          done: 'Icona aggiornata');
    }
  
    @override
    Widget content() => _TabPage(
          children: [
            for (final s in sites)
              Card(
                child: _MenuRow(
                  tile: ListTile(
                  leading: Icon(s['active'] == true ? Icons.language : Icons.public_off),
                  title: Text('${s['name']}'),
                  subtitle: Text('${s['hostname']}  ·  archivio ${s['topic_id']}'
                      '${s['has_icon'] == true ? '  ·  icona propria' : ''}'
                      '${s['active'] == true ? '' : '  ·  disattivato'}'),
                  trailing: IconButton(
                    tooltip: 'Apri il sito in una nuova scheda',
                    icon: const Icon(Icons.open_in_new),
                    onPressed: () => _openSite(s),
                  ),
                ),
                onSelected: (action) async {
                  switch (action) {
                    case 'edit':
                      await _edit(s);
                    case 'icon':
                      await _uploadIcon(s);
                    case 'no-icon':
                      await run(() => api.adminRequest('DELETE', '/sites/${s['hostname']}/icon'));
                  }
                },
                items: const [
                  PopupMenuItem(value: 'edit', child: Text('Modifica')),
                  PopupMenuItem(value: 'icon', child: Text('Carica icona (PNG)')),
                  PopupMenuItem(value: 'no-icon', child: Text('Togli icona')),
                ],
              ),
            ),
        ],
      );
}

/// Sceglie un file PNG dal computer. null = annullato.
Future<Uint8List?> _pickPng() async {
  final input = web.HTMLInputElement()
    ..type = 'file'
    ..accept = 'image/png';
  input.click();
  await input.onChange.first;
  final file = input.files?.item(0);
  if (file == null) return null;
  return (await file.arrayBuffer().toDart).toDart.asUint8List();
}

// ---------------------------------------------------------------- credenziali

class _KeysTab extends StatefulWidget {
  const _KeysTab();

  @override
  State<_KeysTab> createState() => _KeysTabState();
}

class _KeysTabState extends _TabState<_KeysTab> {
  List<_Json> keys = [];
  List<String> topics = [];

  @override
  Future<void> fetch() async {
    keys = List<_Json>.from((await api.adminRequest('GET', '/keys'))['items'] as List);
    topics = await _topicIds(api);
  }

  Future<void> _showKey(String key) => showDialog<void>(
        context: context,
        barrierDismissible: false,
        builder: (ctx) => AlertDialog(
          title: const Text('Chiave generata'),
          content: SizedBox(
            width: 560,
            child: Column(mainAxisSize: MainAxisSize.min, crossAxisAlignment: CrossAxisAlignment.start, children: [
              const Text('Si mostra solo ora: copiala e conservala in modo sicuro.'),
              const SizedBox(height: 12),
              SelectableText(key),
            ]),
          ),
          actions: [
            TextButton(
                onPressed: () => Clipboard.setData(ClipboardData(text: key)), child: const Text('Copia')),
            FilledButton(onPressed: () => Navigator.pop(ctx), child: const Text('Ho salvato la chiave')),
          ],
        ),
      );

  Future<void> _edit([_Json? item]) async {
    final name = TextEditingController(text: item?['name'] as String?);
    final expires = TextEditingController(text: (item?['expires_at'] as String?)?.split('T').first ?? '');
    final selected = Set<String>.from(item?['topics'] as List? ?? const []);
    var active = item?['active'] as bool? ?? true;
    String? created;
    await _formDialog(
      context,
      title: item == null ? 'Nuova credenziale' : 'Credenziale ${item['name']}',
      builder: (setState) => Column(mainAxisSize: MainAxisSize.min, children: [
        _field(name, 'Nome del sistema', enabled: item == null),
        _field(expires, 'Scadenza (facoltativa)', hint: 'AAAA-MM-GG'),
        _chips('Archivi consultabili (la conversione non ne richiede)', topics, selected, setState),
        if (item != null)
          SwitchListTile(
            contentPadding: EdgeInsets.zero,
            title: const Text('Attiva'),
            value: active,
            onChanged: (v) => setState(() => active = v),
          ),
      ]),
      onSave: () => run(() async {
        final body = {'topics': selected.toList(), 'expires': expires.text.trim(), 'active': active};
        if (item == null) {
          final res = await api.adminRequest('POST', '/keys', body: {...body, 'name': name.text});
          created = res['key'] as String;
        } else {
          await api.adminRequest('PUT', '/keys/${item['id']}', body: body);
        }
      }),
    );
    if (created != null && mounted) await _showKey(created!);
  }

  Future<void> _rotate(_Json item) async {
    if (!await _confirm(context, 'Rigenerare la chiave?',
        'La chiave attuale smette subito di funzionare per ${item['name']}.', 'Rigenera')) {
      return;
    }
    String? key;
    await run(() async {
      key = (await api.adminRequest('POST', '/keys/${item['id']}/rotate'))['key'] as String;
    });
    if (key != null && mounted) await _showKey(key!);
  }

  @override
  Widget content() => _TabPage(
        addLabel: 'Nuova credenziale',
        onAdd: _edit,
        children: [
          for (final k in keys)
            Card(
              child: _MenuRow(
                tile: ListTile(
                  leading: Icon(k['active'] == true ? Icons.key_outlined : Icons.key_off_outlined),
                  title: Text('${k['name']}  (${k['key_prefix']}…)'),
                  subtitle: Text('Archivi: ${(k['topics'] as List).isEmpty ? 'nessuno' : (k['topics'] as List).join(', ')}\n'
                      'Scadenza: ${formatDateTime(k['expires_at'] as String?)}  ·  '
                      'Ultimo uso: ${formatDateTime(k['last_used_at'] as String?)}'
                      '${k['active'] == true ? '' : '  ·  revocata'}'),
                  isThreeLine: true,
                ),
                onSelected: (action) async {
                  switch (action) {
                    case 'edit':
                      await _edit(k);
                    case 'rotate':
                      await _rotate(k);
                    default:
                      if (await _confirm(context, 'Eliminare la credenziale?',
                          '${k['name']} perde subito l\'accesso.', 'Elimina')) {
                        await run(() => api.adminRequest('DELETE', '/keys/${k['id']}'));
                      }
                  }
                },
                items: const [
                  PopupMenuItem(value: 'edit', child: Text('Modifica')),
                  PopupMenuItem(value: 'rotate', child: Text('Rigenera la chiave')),
                  PopupMenuItem(value: 'delete', child: Text('Elimina')),
                ],
              ),
            ),
        ],
      );
}

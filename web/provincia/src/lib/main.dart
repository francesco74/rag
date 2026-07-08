import 'dart:convert';
import 'dart:async'; // Aggiunto per il Timer del Progressive Delay
import 'package:flutter/material.dart';
import 'package:http/http.dart' as http;
import 'package:flutter_html/flutter_html.dart'; // For rendering HTML
import 'package:url_launcher/url_launcher.dart'; // For opening links
import 'settings.dart'; // Import the settings file
import 'app_translations.dart'; // Import the translations file
import 'package:flutter/services.dart';
import 'package:tutorial_coach_mark/tutorial_coach_mark.dart';
import 'package:shared_preferences/shared_preferences.dart';
import 'dart:html' as html;

void main() {
  runApp(const AiChatApp());
}

// 1. Define our custom themes
enum AppTheme { light, dark, highContrast }

// 2. Create the Notifier (defaults to light)
final ValueNotifier<AppTheme> themeNotifier = ValueNotifier(AppTheme.light);
final ValueNotifier<AppLang> langNotifier = ValueNotifier(
  _detectBrowserLanguage(),
);

/// Single source of truth for theme cycling.
/// Referenced by both the keyboard shortcut and the AppBar icon button.
void _cycleTheme() {
  if (themeNotifier.value == AppTheme.light) {
    themeNotifier.value = AppTheme.dark;
  } else if (themeNotifier.value == AppTheme.dark) {
    themeNotifier.value = AppTheme.highContrast;
  } else {
    themeNotifier.value = AppTheme.light;
  }
}

// Status enum for feedback
enum FeedbackStatus { none, like, dislike }

class ToggleThemeIntent extends Intent {
  const ToggleThemeIntent();
}

AppLang _detectBrowserLanguage() {
  // Prende la stringa della lingua (es. "it-IT" o "en-US")
  final String browserLang = html.window.navigator.language.toLowerCase();

  if (browserLang.startsWith('it')) {
    return AppLang.it;
  }
  // Default in inglese per tutti gli altri casi
  return AppLang.en;
}

class AiChatApp extends StatelessWidget {
  const AiChatApp({super.key});

  @override
  Widget build(BuildContext context) {
    return ValueListenableBuilder<AppTheme>(
      valueListenable: themeNotifier,
      builder: (_, AppTheme currentTheme, __) {
        return ValueListenableBuilder<AppLang>(
          valueListenable: langNotifier,
          builder: (_, AppLang currentLang, __) {
            final langCode = currentLang == AppLang.it ? 'it' : 'en';
            // Aggiorna l'attributo <html lang="..."> del browser
            html.document.documentElement?.lang = langCode;

            // Define the 3 themes dynamically
            ThemeData activeTheme;
            switch (currentTheme) {
              case AppTheme.light:
                activeTheme = ThemeData(
                  colorScheme: ColorScheme.fromSeed(
                    seedColor: Colors.blue,
                    brightness: Brightness.light,
                  ),
                  textSelectionTheme: TextSelectionThemeData(
                    selectionColor: Colors.blue.withAlpha(70),
                    selectionHandleColor: Colors.blue,
                  ),
                  useMaterial3: true,
                );
                break;
              case AppTheme.dark:
                activeTheme = ThemeData(
                  colorScheme: ColorScheme.fromSeed(
                    seedColor: Colors.blue,
                    brightness: Brightness.dark,
                  ),
                  textSelectionTheme: TextSelectionThemeData(
                    selectionColor: Colors.white.withAlpha(100),
                    selectionHandleColor: Colors.white,
                  ),
                  useMaterial3: true,
                );
                break;
              case AppTheme.highContrast:
                // High contrast uses a pure black background, pure white text, and bright yellow accents
                activeTheme = ThemeData(
                  colorScheme: const ColorScheme.highContrastDark(
                    primary: Colors.yellowAccent,
                    onPrimary: Colors.black,
                    secondary: Colors.cyanAccent,
                    surface: Colors.black,
                    onSurface: Colors.white,
                    error: Colors.redAccent,
                  ),
                  textSelectionTheme: TextSelectionThemeData(
                    selectionColor: Colors.yellowAccent.withAlpha(100),
                    selectionHandleColor: Colors.yellowAccent,
                  ),
                  useMaterial3: true,
                );
                break;
            }

            return Shortcuts(
              shortcuts: <ShortcutActivator, Intent>{
                // Binds Alt + T (or Option + T on Mac) to our Intent
                const SingleActivator(LogicalKeyboardKey.keyT, alt: true):
                    const ToggleThemeIntent(),
              },
              child: Actions(
                actions: <Type, Action<Intent>>{
                  ToggleThemeIntent: CallbackAction<ToggleThemeIntent>(
                    onInvoke: (ToggleThemeIntent intent) {
                      _cycleTheme();
                      return null;
                    },
                  ),
                },
                child: MaterialApp(
                  title: AppSettings.projectName, // Uses your external settings
                  debugShowCheckedModeBanner: false,
                  theme: activeTheme, // Applies the selected theme
                  home: const ChatScreen(),
                ),
              ),
            );
          },
        );
      },
    );
  }
}

// Data model for a chat message
class ChatMessage {
  final String text;
  final bool isUser;
  final String? topic;
  final List<Map<String, dynamic>> sources;
  final bool isError;
  final bool isSystemMessage; // Exclude from history sent to LLM
  final String role; // 'user' or 'model'

  // Store feedback state
  FeedbackStatus feedback;
  bool isSourcesExpanded;

  ChatMessage({
    required this.text,
    this.isUser = false,
    this.topic,
    this.sources = const [],
    this.isError = false,
    this.isSystemMessage = false,
    this.isSourcesExpanded = false,
    this.feedback = FeedbackStatus.none,
  }) : role = isUser ? 'user' : 'model';
}

class ChatScreen extends StatefulWidget {
  const ChatScreen({super.key});

  @override
  State<ChatScreen> createState() => _ChatScreenState();
}

Widget _buildTutorialText(String title, String desc) {
  return Column(
    mainAxisSize: MainAxisSize.min,
    crossAxisAlignment: CrossAxisAlignment.start,
    children: [
      Text(
        title,
        style: const TextStyle(
          fontWeight: FontWeight.bold,
          color: Colors.white,
          fontSize: 20,
        ),
      ),
      Padding(
        padding: const EdgeInsets.only(top: 10.0),
        child: Text(desc, style: const TextStyle(color: Colors.white)),
      ),
    ],
  );
}

class _ChatScreenState extends State<ChatScreen> {
  final TextEditingController _textController = TextEditingController();
  final ScrollController _scrollController = ScrollController();
  final List<ChatMessage> _messages = [];
  bool _isLoading = false;
  bool _isMaintenanceMode = false;
  final FocusNode _textFocusNode = FocusNode();

  final GlobalKey _filterKey = GlobalKey();
  final GlobalKey _langKey = GlobalKey();
  final GlobalKey _themeKey = GlobalKey();
  final GlobalKey _deleteKey = GlobalKey();
  final GlobalKey _sendKey = GlobalKey();

  TutorialCoachMark? _tutorialCoachMark;
  List<TargetFocus> _targets = [];

  Map<String, String> _subTopicDescriptions = {};
  List<String> _availableSubTopicIds = [];
  Set<String> _selectedSubTopics = {};
  bool _allowSubtopicSelection = false;

  // --- STATO PER PROGRESSIVE DELAY ---
  Timer? _longWaitTimer;
  String? _dynamicLoadingMessage;

  @override
  void initState() {
    super.initState();
    _addWelcomeMessage();
    _fetchConfig();

    WidgetsBinding.instance.addPostFrameCallback((_) {
      _checkFirstRun();
    });
  }

  Future<void> _checkFirstRun() async {
    final prefs = await SharedPreferences.getInstance();
    bool isFirstRun = prefs.getBool('tutorial_seen') ?? false;

    if (!isFirstRun) {
      await Future.delayed(const Duration(milliseconds: 500));
      _showTutorial();
      await prefs.setBool('tutorial_seen', true);
    } else {
      debugPrint("Tutorial già visto, non mostro di nuovo.");
    }
  }

  void _initTutorialTargets() {
    _targets.clear();

    // 1. Filtro Serie
    _targets.add(
      TargetFocus(
        identify: "FilterTarget",
        keyTarget: _filterKey,
        alignSkip: Alignment.topRight,
        contents: [
          TargetContent(
            align: ContentAlign.bottom,
            builder: (context, controller) => _buildTutorialText(
              AppTranslations.get('filter_subtopics', langNotifier.value),
              AppTranslations.get('filter_subtopics_desc', langNotifier.value),
            ),
          ),
        ],
      ),
    );

    // 2. Cambio Lingua
    _targets.add(
      TargetFocus(
        identify: "LangTarget",
        keyTarget: _langKey,
        contents: [
          TargetContent(
            align: ContentAlign.bottom,
            builder: (context, controller) => _buildTutorialText(
              AppTranslations.get('change_language', langNotifier.value),
              AppTranslations.get('change_language_desc', langNotifier.value),
            ),
          ),
        ],
      ),
    );

    // 3. Tema
    _targets.add(
      TargetFocus(
        identify: "ThemeTarget",
        keyTarget: _themeKey,
        contents: [
          TargetContent(
            align: ContentAlign.bottom,
            builder: (context, controller) => _buildTutorialText(
              AppTranslations.get('change_theme', langNotifier.value),
              AppTranslations.get('change_theme_desc', langNotifier.value),
            ),
          ),
        ],
      ),
    );

    _targets.add(
      TargetFocus(
        identify: "DeleteTarget",
        keyTarget: _deleteKey,
        contents: [
          TargetContent(
            align: ContentAlign.bottom,
            builder: (context, controller) => _buildTutorialText(
              AppTranslations.get('delete_chat', langNotifier.value),
              AppTranslations.get('delete_chat_desc', langNotifier.value),
            ),
          ),
        ],
      ),
    );

    _targets.add(
      TargetFocus(
        identify: "SendTarget",
        keyTarget: _sendKey,
        contents: [
          TargetContent(
            align: ContentAlign.bottom,
            builder: (context, controller) => _buildTutorialText(
              AppTranslations.get('send_question', langNotifier.value),
              AppTranslations.get('send_question_desc', langNotifier.value),
            ),
          ),
        ],
      ),
    );
  }

  void _showTutorial() {
    _initTutorialTargets();
    _tutorialCoachMark = TutorialCoachMark(
      targets: _targets,
      colorShadow: Colors.black,
      opacityShadow: 0.85,
      paddingFocus: 10,
      textSkip: AppTranslations.get('skip', langNotifier.value).toUpperCase(),
      onClickTarget: (target) =>
          debugPrint("Target cliccato: ${target.identify}"),
    )..show(context: context);
  }

  Future<void> _fetchConfig() async {
    try {
      final response = await http.post(
        Uri.parse("${AppSettings.apiUrl}/config"),
        headers: {
          "Content-Type": "application/json",
          "Authorization": "Bearer ${AppSettings.apiSecretKeyValue}",
        },
        body: jsonEncode({"topic_id": AppSettings.getTopicId}),
      );

      if (response.statusCode == 200) {
        final data = jsonDecode(response.body);
        final List rawSubTopics = data['sub_topics'];

        setState(() {
          _allowSubtopicSelection = data['allow_subtopic_selection'] ?? false;
          _availableSubTopicIds = rawSubTopics
              .map((item) => item['id'].toString())
              .toList();
          _subTopicDescriptions = {
            for (var item in rawSubTopics)
              item['id'].toString(): item['desc'].toString(),
          };

          _selectedSubTopics = _availableSubTopicIds.toSet();
          _isMaintenanceMode = false;
        });
      } else {
        setState(() {
          _isMaintenanceMode = true;
        });
      }
    } catch (e) {
      debugPrint("Errore critico configurazione: $e");
      setState(() {
        _isMaintenanceMode = true;
      });
    }
  }

  /// Adds the initial system welcome message
  void _addWelcomeMessage() {
    _messages.insert(
      0,
      ChatMessage(
        text: AppTranslations.get('welcome_msg', langNotifier.value),
        isSystemMessage: true,
      ),
    );
  }

  Future<void> _promptFeedbackComment(ChatMessage message, bool isLike) async {
    if (message.isUser || message.isError || message.isSystemMessage) return;

    setState(() {
      message.feedback = isLike ? FeedbackStatus.like : FeedbackStatus.dislike;
    });

    final TextEditingController commentController = TextEditingController();

    await showDialog(
      context: context,
      builder: (context) {
        return AlertDialog(
          title: Text(
            isLike
                ? AppTranslations.get('like_msg', langNotifier.value)
                : AppTranslations.get('dislike_msg', langNotifier.value),
          ),
          content: TextField(
            controller: commentController,
            maxLines: 3,
            decoration: InputDecoration(
              hintText: AppTranslations.get('comment_hint', langNotifier.value),
              border: OutlineInputBorder(
                borderRadius: BorderRadius.circular(8.0),
              ),
            ),
          ),
          actions: [
            TextButton(
              onPressed: () => Navigator.of(context).pop(),
              child: Text(AppTranslations.get('skip', langNotifier.value)),
            ),
            ElevatedButton(
              onPressed: () => Navigator.of(context).pop(),
              child: Text(AppTranslations.get('submit', langNotifier.value)),
            ),
          ],
        );
      },
    );

    await _sendFeedback(message, isLike, commentController.text.trim());
  }

  Widget _buildMaintenancePage() {
    final theme = Theme.of(context);
    return Scaffold(
      body: Center(
        child: Padding(
          padding: const EdgeInsets.all(32.0),
          child: Column(
            mainAxisAlignment: MainAxisAlignment.center,
            children: [
              Icon(
                Icons.construction,
                size: 80,
                color: theme.colorScheme.primary,
              ),
              const SizedBox(height: 24),
              Text(
                AppTranslations.get('maintenance_mode', langNotifier.value),
                style: theme.textTheme.headlineMedium!.copyWith(
                  fontWeight: FontWeight.bold,
                ),
                textAlign: TextAlign.center,
              ),
              const SizedBox(height: 16),
              Text(
                AppTranslations.get('maintenance_msg', langNotifier.value),
                style: theme.textTheme.bodyLarge,
                textAlign: TextAlign.center,
              ),
              const SizedBox(height: 32),
              ElevatedButton.icon(
                onPressed: () {
                  setState(() => _isMaintenanceMode = false);
                  _fetchConfig(); // Riprova la configurazione
                },
                icon: const Icon(Icons.refresh),
                label: Text(
                  AppTranslations.get('retry_connection', langNotifier.value),
                ),
              ),
            ],
          ),
        ),
      ),
    );
  }

  /// Builds the static project information header at the top of the chat
  Widget _buildProjectInfoHeader() {
    final theme = Theme.of(context);

    return Container(
      margin: const EdgeInsets.only(bottom: 24.0, top: 16.0),
      decoration: BoxDecoration(
        color: theme.colorScheme.surfaceContainerHighest.withAlpha(128),
        borderRadius: BorderRadius.circular(16.0),
        border: Border.all(color: theme.colorScheme.outlineVariant, width: 1),
      ),
      child: FutureBuilder<String>(
        future: DefaultAssetBundle.of(context).loadString(
          AppTranslations.get('welcome_html_path', langNotifier.value),
        ),
        builder: (context, snapshot) {
          if (snapshot.connectionState == ConnectionState.waiting) {
            return const Padding(
              padding: EdgeInsets.all(48.0),
              child: Center(child: CircularProgressIndicator()),
            );
          }
          if (snapshot.hasError || !snapshot.hasData) {
            return Padding(
              padding: const EdgeInsets.all(24.0),
              child: Text(
                AppTranslations.get('welcome_load_error', langNotifier.value),
                style: theme.textTheme.bodyMedium!.copyWith(
                  color: theme.colorScheme.error,
                ),
                textAlign: TextAlign.center,
              ),
            );
          }

          return Html(
            data: snapshot.data,
            extensions: [
              TagExtension(
                tagsToExtend: {"a"},
                builder: (extensionContext) {
                  final url = extensionContext.attributes['href'];
                  final text = extensionContext.element?.text ?? 'Link';

                  if (extensionContext.classes.contains("btn")) {
                    return Padding(
                      padding: const EdgeInsets.symmetric(
                        horizontal: 4.0,
                        vertical: 8.0,
                      ),
                      child: OutlinedButton(
                        style: OutlinedButton.styleFrom(
                          side: BorderSide(
                            color: theme.colorScheme.primary,
                            width: 1.5,
                          ),
                          foregroundColor: theme.colorScheme.primary,
                          padding: const EdgeInsets.symmetric(
                            horizontal: 20.0,
                            vertical: 12.0,
                          ),
                        ),
                        onPressed: () {
                          if (url != null) _launchUrl(url);
                        },
                        child: Text(
                          text,
                          style: const TextStyle(fontWeight: FontWeight.bold),
                        ),
                      ),
                    );
                  }

                  return Semantics(
                    link: true,
                    child: InkWell(
                      onTap: () {
                        if (url != null) _launchUrl(url);
                      },
                      child: Text(
                        text,
                        style: TextStyle(
                          color: theme.colorScheme.primary,
                          decoration: TextDecoration.underline,
                        ),
                      ),
                    ),
                  );
                },
              ),
            ],
            style: {
              "body": Style(
                margin: Margins.zero,
                padding: HtmlPaddings.all(24.0),
                backgroundColor: Colors.transparent,
                color: theme.colorScheme.onSurface,
                fontFamily: 'sans-serif',
                textAlign: TextAlign.center,
              ),
              ".icon": Style(
                fontSize: FontSize(48.0),
                margin: Margins.only(bottom: 16.0),
              ),
              "h1": Style(
                fontSize: FontSize(22.0),
                fontWeight: FontWeight.bold,
                color: theme.colorScheme.primary,
                margin: Margins.only(bottom: 4.0),
              ),
              "p": Style(
                fontSize: FontSize(16.0),
                color: theme.colorScheme.onSurfaceVariant,
                margin: Margins.only(bottom: 20.0),
              ),
              ".links": Style(
                textAlign: TextAlign.center,
                margin: Margins.only(bottom: 16.0),
              ),
              ".divider": Style(
                height: Height(1.0),
                backgroundColor: theme.colorScheme.outlineVariant,
                margin: Margins.symmetric(vertical: 20.0),
              ),
              ".disclaimer": Style(
                backgroundColor: theme.colorScheme.secondaryContainer,
                padding: HtmlPaddings.all(16.0),
                border: Border(
                  left: BorderSide(
                    color: theme.colorScheme.secondary,
                    width: 4.0,
                  ),
                ),
                margin: Margins.only(top: 16.0),
              ),
              ".disclaimer-title": Style(
                color: theme.colorScheme.onSecondaryContainer,
                fontWeight: FontWeight.bold,
                fontSize: FontSize(16.0),
                margin: Margins.only(bottom: 8.0),
              ),
              ".disclaimer-text": Style(
                fontSize: FontSize(14.0),
                color: theme.colorScheme.onSecondaryContainer,
              ),
              ".project": Style(
                fontSize: FontSize(13.0),
                fontStyle: FontStyle.italic,
                color: theme.colorScheme.onErrorContainer.withAlpha(200),
                textAlign: TextAlign.center,
                margin: Margins.only(bottom: 4.0),
              ),
            },
          );
        },
      ),
    );
  }

  @override
  void dispose() {
    _textController.dispose();
    _scrollController.dispose();
    _textFocusNode.dispose();
    _longWaitTimer?.cancel(); // Prevenzione Memory Leak
    super.dispose();
  }

  // ===========================================================================
  // CORE API LOGIC
  // ===========================================================================

  /// Polls the status endpoint until the task is completed or failed
  Future<Map<String, dynamic>> _pollForResponse(String taskId) async {
    final String statusUrl = "${AppSettings.apiUrl}/status/$taskId";

    final DateTime startTime = DateTime.now();
    const int timeoutSeconds = 180;

    int consecutiveErrors = 0;
    const int maxErrors = 3;

    while (true) {
      if (DateTime.now().difference(startTime).inSeconds > timeoutSeconds) {
        throw Exception(AppTranslations.get('timeout', langNotifier.value));
      }

      try {
        final response = await http.get(
          Uri.parse(statusUrl),
          headers: {"Authorization": "Bearer ${AppSettings.apiSecretKeyValue}"},
        );

        // FIX: Accettiamo 200 e 202. La UI non va in crash durante l'elaborazione.
        if (response.statusCode == 200 || response.statusCode == 202) {
          consecutiveErrors =
              0; // Azzera gli errori solo su reale successo di rete

          final data = jsonDecode(utf8.decode(response.bodyBytes));
          final String status = data['status'];

          if (status == 'completed' || status == 'success') {
            return data['data'] ?? data;
          } else if (status == 'failed') {
            return {
              'status': 'failed',
              'error': data['error'] ?? "Worker error",
            };
          }
        } else {
          // Errori severi 500, 502, etc.
          consecutiveErrors++;
        }
      } catch (e) {
        consecutiveErrors++;
      }

      if (consecutiveErrors >= maxErrors) {
        throw Exception(
          AppTranslations.get('connection_lost', langNotifier.value),
        );
      }

      await Future.delayed(const Duration(seconds: 2));
    }
  }

  /// Sends the user's query and chat history to the Flask backend
  Future<void> _handleSendPressed() async {
    final text = _textController.text;
    if (text.isEmpty) return;

    final userMessage = ChatMessage(text: text, isUser: true);
    _addMessage(userMessage);
    _textController.clear();

    setState(() {
      _isLoading = true;
      _dynamicLoadingMessage = null; // Reset per la nuova chiamata
    });

    // --- PROGRESSIVE DELAY MESSAGING ---
    _longWaitTimer?.cancel();
    _longWaitTimer = Timer(const Duration(seconds: 20), () {
      if (mounted && _isLoading) {
        setState(() {
          // Assicurati che 'taking_longer' sia tradotto nel tuo file app_translations
          _dynamicLoadingMessage = AppTranslations.get(
            'taking_longer',
            langNotifier.value,
          );
        });
      }
    });

    final chatHistory = _messages
        .where((msg) => !msg.isError && !msg.isSystemMessage)
        .toList()
        .reversed
        .map((msg) => {'role': msg.role, 'text': msg.text})
        .toList();

    final List<Map<String, dynamic>> chatHistoryForApi = chatHistory.length > 1
        ? chatHistory.sublist(0, chatHistory.length - 1)
        : [];

    final lastQuery = text;

    try {
      final String chatUrl = "${AppSettings.apiUrl}/chat";

      final response = await http
          .post(
            Uri.parse(chatUrl),
            headers: {
              "Content-Type": "application/json",
              "Authorization": "Bearer ${AppSettings.apiSecretKeyValue}",
            },
            body: jsonEncode({
              "history": chatHistoryForApi,
              "query": lastQuery,
              "sub_topics": _selectedSubTopics.toList(),
              "topic_id": AppSettings.getTopicId,
            }),
          )
          .timeout(const Duration(seconds: 10));

      if (response.statusCode == 202) {
        final data = jsonDecode(utf8.decode(response.bodyBytes));
        final String taskId = data['task_id'];

        final resultData = await _pollForResponse(taskId);

        if (resultData['status'] == 'failed' ||
            resultData['status'] == 'not_found') {
          _addMessage(
            ChatMessage(
              text:
                  "<p><strong>${AppTranslations.get('error', langNotifier.value)}:</strong> ${resultData['error'] ?? resultData['message']}</p>",
              isError: true,
              isSystemMessage: true,
            ),
          );
        } else {
          _addMessage(
            ChatMessage(
              text: resultData['answer'],
              topic: resultData['topic'],
              sources: (resultData['sources'] as List)
                  .map((s) => s as Map<String, dynamic>)
                  .toList(),
            ),
          );
        }
      } else {
        final data = jsonDecode(utf8.decode(response.bodyBytes));
        _addMessage(
          ChatMessage(
            text:
                "<p>${AppTranslations.get('error', langNotifier.value)} (${response.statusCode}): ${data['message'] ?? AppTranslations.get('unknown_error', langNotifier.value)}</p>",
            isError: true,
            isSystemMessage: true,
          ),
        );
      }
    } catch (e) {
      _addMessage(
        ChatMessage(
          text:
              "<p>${AppTranslations.get('connection_lost', langNotifier.value)}</p>",
          isError: true,
          isSystemMessage: true,
        ),
      );
    } finally {
      // Spegne SEMPRE il timer, a prescindere dal successo o dal fallimento
      _longWaitTimer?.cancel();

      if (mounted) {
        setState(() {
          _isLoading = false;
        });
        _textFocusNode.requestFocus();
      }
    }
  }

  // --- Feedback Logic ---
  Future<void> _sendFeedback(
    ChatMessage message,
    bool isLike,
    String comment,
  ) async {
    if (message.isUser || message.isError || message.isSystemMessage) return;

    int msgIndex = _messages.indexOf(message);
    String userQuery = "";

    if (msgIndex != -1 && msgIndex + 1 < _messages.length) {
      userQuery = _messages[msgIndex + 1].text;
    }

    List<Map<String, dynamic>> contextHistory = [];
    if (msgIndex + 2 < _messages.length) {
      contextHistory = _messages
          .sublist(msgIndex + 2)
          .where((m) => !m.isError && !m.isSystemMessage)
          .toList()
          .reversed
          .map((m) => {'role': m.role, 'text': m.text})
          .toList();
    }

    setState(() {
      message.feedback = isLike ? FeedbackStatus.like : FeedbackStatus.dislike;
    });

    try {
      final String apiUrl = "${AppSettings.apiUrl}/feedback";
      final feedbackResponse = await http.post(
        Uri.parse(apiUrl),
        headers: {
          "Content-Type": "application/json",
          "Authorization": "Bearer ${AppSettings.apiSecretKeyValue}",
        },
        body: jsonEncode({
          "query": userQuery,
          "answer": message.text,
          "topic_id": message.topic,
          "rating": isLike ? 1 : -1,
          "history": contextHistory,
          "comment": comment,
        }),
      );

      if (feedbackResponse.statusCode < 200 ||
          feedbackResponse.statusCode >= 300) {
        throw Exception("Server returned ${feedbackResponse.statusCode}");
      }

      if (mounted) {
        ScaffoldMessenger.of(context).showSnackBar(
          SnackBar(
            content: Text(
              AppTranslations.get('feedback_received', langNotifier.value),
            ),
            duration: const Duration(seconds: 4),
          ),
        );
      }
    } catch (e) {
      if (mounted) {
        ScaffoldMessenger.of(context).showSnackBar(
          SnackBar(
            content: Text(
              AppTranslations.get('feedback_error', langNotifier.value),
            ),
            duration: const Duration(seconds: 4),
          ),
        );
      }
    }
  }

  /// Adds a new message to the list and scrolls to the bottom
  void _addMessage(ChatMessage message) {
    setState(() {
      _messages.insert(0, message);
    });
    if (_scrollController.hasClients) {
      _scrollController.animateTo(
        0.0,
        duration: const Duration(milliseconds: 300),
        curve: Curves.easeOut,
      );
    }
  }

  /// --- Opens Nginx Document URLs ---
  Future<void> _launchUrl(String urlString) async {
    final Uri url = Uri.parse(urlString);
    if (!await launchUrl(url, mode: LaunchMode.externalApplication)) {
      _addMessage(
        ChatMessage(
          text:
              "${AppTranslations.get('document_error', langNotifier.value)} ($urlString)",
          isError: true,
          isSystemMessage: true,
        ),
      );
    }
  }

  // --- Clears the chat history ---
  void _clearChat() {
    setState(() {
      _messages.clear();
      _addWelcomeMessage();
    });

    ScaffoldMessenger.of(context).showSnackBar(
      SnackBar(
        content: Text(AppTranslations.get('chat_cleared', langNotifier.value)),
        duration: const Duration(seconds: 3),
      ),
    );
  }

  void _showSubTopicSelector() {
    showModalBottomSheet(
      context: context,
      builder: (context) {
        return StatefulBuilder(
          builder: (context, setModalState) {
            return Container(
              padding: const EdgeInsets.all(16.0),
              child: Column(
                mainAxisSize: MainAxisSize.min,
                children: [
                  Text(
                    AppTranslations.get('subtopics_title', langNotifier.value),
                    style: Theme.of(context).textTheme.titleLarge,
                  ),
                  const Divider(),
                  Expanded(
                    child: ListView(
                      children: _availableSubTopicIds.map((id) {
                        return CheckboxListTile(
                          title: Text(_subTopicDescriptions[id] ?? id),
                          subtitle: Text(
                            id,
                            style: const TextStyle(fontSize: 10),
                          ),
                          value: _selectedSubTopics.contains(id),
                          onChanged: (bool? value) {
                            setState(() {
                              if (value == true) {
                                _selectedSubTopics.add(id);
                              } else {
                                _selectedSubTopics.remove(id);
                              }
                            });
                            setModalState(() {});
                          },
                        );
                      }).toList(),
                    ),
                  ),
                ],
              ),
            );
          },
        );
      },
    );
  }

  // ===========================================================================
  // UI BUILDING
  // ===========================================================================

  @override
  Widget build(BuildContext context) {
    if (_isMaintenanceMode) {
      return _buildMaintenancePage();
    }

    return Scaffold(
      appBar: AppBar(
        title: Text(AppSettings.projectName),
        actions: [
          if (_allowSubtopicSelection && _availableSubTopicIds.isNotEmpty)
            IconButton(
              key: _filterKey,
              icon: const Icon(Icons.filter_list),
              onPressed: _showSubTopicSelector,
              tooltip: AppTranslations.get(
                'filter_subtopics',
                langNotifier.value,
              ),
            ),
          ValueListenableBuilder<AppLang>(
            valueListenable: langNotifier,
            builder: (_, AppLang currentLang, __) {
              return TextButton(
                key: _langKey,
                onPressed: () {
                  langNotifier.value = currentLang == AppLang.it
                      ? AppLang.en
                      : AppLang.it;
                  _clearChat();
                },
                child: Text(
                  currentLang == AppLang.it ? "🇮🇹 IT" : "🇬🇧 EN",
                  style: TextStyle(
                    fontSize: 16,
                    color: Theme.of(context).colorScheme.onSurface,
                  ),
                ),
              );
            },
          ),

          ValueListenableBuilder<AppTheme>(
            valueListenable: themeNotifier,
            builder: (_, AppTheme currentTheme, __) {
              IconData themeIcon;
              if (currentTheme == AppTheme.light) {
                themeIcon = Icons.dark_mode;
              } else if (currentTheme == AppTheme.dark) {
                themeIcon = Icons.contrast;
              } else {
                themeIcon = Icons.light_mode;
              }

              return IconButton(
                key: _themeKey,
                icon: Icon(themeIcon),
                tooltip: AppTranslations.get(
                  'toggle_theme',
                  langNotifier.value,
                ),
                onPressed: _cycleTheme,
              );
            },
          ),

          IconButton(
            key: _deleteKey,
            icon: const Icon(Icons.delete_sweep_outlined),
            tooltip: AppTranslations.get('clear_chat', langNotifier.value),
            onPressed: _isLoading ? null : _clearChat,
          ),
          const SizedBox(width: 8),
        ],
      ),
      body: Column(
        children: [
          Expanded(
            child: ListView.builder(
              controller: _scrollController,
              reverse: true,
              padding: const EdgeInsets.all(16.0),
              itemCount: _messages.length + 1,
              itemBuilder: (context, index) {
                if (index == _messages.length) {
                  return _buildProjectInfoHeader();
                }
                return _buildMessageBubble(_messages[index]);
              },
            ),
          ),
          // Sostituisce il vecchio LinearProgressIndicator con l'indicatore testuale
          if (_isLoading) _buildTypingIndicator(),

          _buildTextInputArea(),
        ],
      ),
    );
  }

  /// Mostra un feedback visivo chiaro e testuale durante il polling.
  /// Mantiene pulita la lista dei messaggi _messages.
  Widget _buildTypingIndicator() {
    final theme = Theme.of(context);

    // Usa il messaggio dinamico dal Timer, oppure il fallback standard 'ai_processing'
    final displayText =
        _dynamicLoadingMessage ??
        AppTranslations.get('ai_processing', langNotifier.value);

    return Padding(
      padding: const EdgeInsets.symmetric(horizontal: 24.0, vertical: 16.0),
      child: Row(
        mainAxisSize: MainAxisSize.min, // Evita che occupi tutta la larghezza
        children: [
          SizedBox(
            width: 16,
            height: 16,
            child: CircularProgressIndicator(
              strokeWidth: 2.5,
              color: theme.colorScheme.primary,
            ),
          ),
          const SizedBox(width: 16),
          // Semantics permette allo screen reader di annunciare l'attesa
          Semantics(
            liveRegion: true,
            child: Text(
              displayText,
              style: theme.textTheme.bodyMedium!.copyWith(
                color: theme.colorScheme.onSurfaceVariant,
                fontStyle: FontStyle.italic,
              ),
            ),
          ),
        ],
      ),
    );
  }

  /// Builds the chat bubble for a given message
  Widget _buildMessageBubble(ChatMessage message) {
    final bool isUser = message.isUser;
    final theme = Theme.of(context);

    Widget messageContent;

    if (isUser) {
      messageContent = SelectableText(
        message.text,
        style: theme.textTheme.bodyLarge!.copyWith(
          color: theme.colorScheme.onPrimary,
        ),
      );
    } else {
      messageContent = SelectionArea(
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            Html(
              data: message.text,
              style: {
                "body": Style(margin: Margins.zero, padding: HtmlPaddings.zero),
                "p": Style(
                  fontSize: FontSize(
                    Theme.of(context).textTheme.bodyLarge!.fontSize!,
                  ),
                  color: message.isError
                      ? theme.colorScheme.onErrorContainer
                      : theme.colorScheme.onSurface,
                ),
                "ul": Style(
                  padding: HtmlPaddings.only(left: 20),
                  color: message.isError
                      ? theme.colorScheme.onErrorContainer
                      : theme.colorScheme.onSurface,
                ),
                "li": Style(
                  color: message.isError
                      ? theme.colorScheme.onErrorContainer
                      : theme.colorScheme.onSurface,
                ),
              },
            ),
            _buildSources(message),

            if (!message.isError && !message.isSystemMessage)
              Padding(
                padding: const EdgeInsets.only(top: 8.0),
                child: Row(
                  mainAxisSize: MainAxisSize.min,
                  children: [
                    IconButton(
                      icon: Icon(
                        message.feedback == FeedbackStatus.like
                            ? Icons.thumb_up
                            : Icons.thumb_up_outlined,
                        size: 20,
                        color: message.feedback == FeedbackStatus.like
                            ? Colors.green
                            : theme.colorScheme.onSurfaceVariant,
                      ),
                      onPressed: () => _promptFeedbackComment(message, true),
                      tooltip: AppTranslations.get(
                        'good_response',
                        langNotifier.value,
                      ),
                    ),
                    IconButton(
                      icon: Icon(
                        message.feedback == FeedbackStatus.dislike
                            ? Icons.thumb_down
                            : Icons.thumb_down_outlined,
                        size: 20,
                        color: message.feedback == FeedbackStatus.dislike
                            ? Colors.red
                            : theme.colorScheme.onSurfaceVariant,
                      ),
                      onPressed: () => _promptFeedbackComment(message, false),
                      tooltip: AppTranslations.get(
                        'bad_response',
                        langNotifier.value,
                      ),
                    ),
                  ],
                ),
              ),
          ],
        ),
      );
    }

    return Semantics(
      liveRegion: true,
      child: Container(
        margin: const EdgeInsets.symmetric(vertical: 8.0),
        child: Row(
          mainAxisAlignment: isUser
              ? MainAxisAlignment.end
              : MainAxisAlignment.start,
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            if (!isUser)
              CircleAvatar(
                backgroundColor: theme.colorScheme.secondary,
                child: Icon(
                  message.isError
                      ? Icons.error_outline
                      : (message.isSystemMessage
                            ? Icons.info_outline
                            : Icons.computer),
                  color: theme.colorScheme.onSecondary,
                ),
              ),
            if (isUser) const SizedBox(width: 40),
            Expanded(
              child: Container(
                margin: const EdgeInsets.symmetric(horizontal: 12.0),
                padding: const EdgeInsets.all(16.0),
                decoration: BoxDecoration(
                  color: isUser
                      ? theme.colorScheme.primary
                      : (message.isError
                            ? theme.colorScheme.errorContainer
                            : theme.colorScheme.surfaceContainerHighest),
                  borderRadius: BorderRadius.only(
                    topLeft: const Radius.circular(20),
                    topRight: const Radius.circular(20),
                    bottomLeft: isUser
                        ? const Radius.circular(20)
                        : Radius.zero,
                    bottomRight: isUser
                        ? Radius.zero
                        : const Radius.circular(20),
                  ),
                ),
                child: messageContent,
              ),
            ),
            if (isUser)
              CircleAvatar(
                backgroundColor: theme.colorScheme.primary,
                child: Icon(Icons.person, color: theme.colorScheme.onPrimary),
              ),
            if (!isUser) const SizedBox(width: 40),
          ],
        ),
      ),
    );
  }

  /// Builds the source chips for an AI message (Expandable)
  Widget _buildSources(ChatMessage message) {
    if (message.isUser || message.sources.isEmpty) {
      return const SizedBox.shrink();
    }

    final theme = Theme.of(context);
    final String topic =
        message.topic ??
        AppTranslations.get('unknown_topic', langNotifier.value);

    return Padding(
      padding: const EdgeInsets.only(top: 12.0),
      child: StatefulBuilder(
        builder: (context, setLocalState) {
          return Column(
            crossAxisAlignment: CrossAxisAlignment.start,
            children: [
              MouseRegion(
                cursor: SystemMouseCursors.click,
                child: GestureDetector(
                  onTap: () {
                    setLocalState(() {
                      message.isSourcesExpanded = !message.isSourcesExpanded;
                    });
                  },
                  child: Row(
                    mainAxisSize: MainAxisSize.min,
                    children: [
                      Icon(
                        message.isSourcesExpanded
                            ? Icons.keyboard_arrow_down
                            : Icons.keyboard_arrow_right,
                        size: 20,
                        color: theme.colorScheme.onSurface.withAlpha(204),
                      ),
                      const SizedBox(width: 4.0),
                      Text(
                        "${AppTranslations.get('sources', langNotifier.value)} ${message.sources.length}",
                        style: theme.textTheme.bodySmall!.copyWith(
                          fontWeight: FontWeight.bold,
                          color: theme.colorScheme.onSurface.withAlpha(204),
                        ),
                      ),
                    ],
                  ),
                ),
              ),

              AnimatedSize(
                duration: const Duration(milliseconds: 250),
                curve: Curves.easeInOut,
                alignment: Alignment.topCenter,
                child: message.isSourcesExpanded
                    ? Padding(
                        padding: const EdgeInsets.only(top: 8.0, left: 8.0),
                        child: Wrap(
                          spacing: 8.0,
                          runSpacing: 4.0,
                          children: message.sources.map((source) {
                            final String fileName =
                                source['file'] ??
                                AppTranslations.get(
                                  'unknown_file',
                                  langNotifier.value,
                                );
                            final String subTopic = source['sub_topic'] ?? '';

                            final String url = subTopic.isNotEmpty
                                ? "${AppSettings.downloadDocumentUrl}/$topic/$subTopic/$fileName"
                                : "${AppSettings.downloadDocumentUrl}/$topic/$fileName";

                            return ActionChip(
                              onPressed: () => _launchUrl(url),
                              avatar: Icon(
                                Icons.link,
                                size: 16,
                                color: theme.colorScheme.onSecondaryContainer,
                              ),
                              label: Text(fileName),
                              labelStyle: theme.textTheme.labelSmall?.copyWith(
                                color: theme.colorScheme.onSecondaryContainer,
                                fontWeight: FontWeight.bold,
                              ),
                              backgroundColor:
                                  theme.colorScheme.secondaryContainer,
                              labelPadding: const EdgeInsets.symmetric(
                                horizontal: 8.0,
                              ),
                              visualDensity: VisualDensity.compact,
                              side: BorderSide(
                                color: theme.colorScheme.onSecondaryContainer
                                    .withAlpha(50),
                              ),
                            );
                          }).toList(),
                        ),
                      )
                    : const SizedBox.shrink(),
              ),
            ],
          );
        },
      ),
    );
  }

  /// Builds the bottom text input field and send button
  Widget _buildTextInputArea() {
    final theme = Theme.of(context);
    return Container(
      padding: const EdgeInsets.all(16.0),
      decoration: BoxDecoration(
        color: theme.colorScheme.surface,
        boxShadow: [
          BoxShadow(
            color: Colors.black.withValues(alpha: 0.1),
            blurRadius: 10,
            offset: const Offset(0, -5),
          ),
        ],
      ),
      child: SafeArea(
        child: Row(
          children: [
            Expanded(
              child: TextField(
                controller: _textController,
                focusNode: _textFocusNode,
                enabled: !_isLoading,
                autocorrect: true,
                enableSuggestions: true,
                decoration: InputDecoration(
                  hintText: AppTranslations.get('ask_hint', langNotifier.value),
                  filled: true,
                  fillColor: theme.colorScheme.surfaceContainerHighest,
                  border: OutlineInputBorder(
                    borderRadius: BorderRadius.circular(30.0),
                    borderSide: BorderSide.none,
                  ),
                  contentPadding: const EdgeInsets.symmetric(
                    horizontal: 20.0,
                    vertical: 16.0,
                  ),
                ),
                onSubmitted: (_) => _handleSendPressed(),
              ),
            ),
            const SizedBox(width: 12.0),
            FloatingActionButton(
              key: _sendKey,
              onPressed: _isLoading ? null : _handleSendPressed,
              backgroundColor: theme.colorScheme.primary,
              tooltip: AppTranslations.get('send_question', langNotifier.value),
              child: Icon(Icons.send, color: theme.colorScheme.onPrimary),
            ),
          ],
        ),
      ),
    );
  }
}

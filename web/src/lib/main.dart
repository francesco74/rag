/*
 * ====================================================================
 * RAG AI Chat App (Flutter Client - Web Only)
 * ====================================================================
 */

import 'dart:convert';
import 'package:flutter/material.dart';
import 'package:http/http.dart' as http;
import 'package:flutter_html/flutter_html.dart'; // For rendering HTML
import 'package:url_launcher/url_launcher.dart'; // For opening links
import 'settings.dart'; // Import the settings file

void main() {
  runApp(const AiChatApp());
}

class AiChatApp extends StatelessWidget {
  const AiChatApp({super.key});

  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      title: 'AI Chat',
      debugShowCheckedModeBanner: false,
      theme: ThemeData(
        colorScheme: ColorScheme.fromSeed(
          seedColor: Colors.blue,
          brightness: Brightness.dark,
        ),
        // Define a contrasting selection color for dark theme
        // This ensures selected text is visible against the bubble background
        textSelectionTheme: TextSelectionThemeData(
          selectionColor: Colors.white.withAlpha(100),
          selectionHandleColor: Colors.white,
        ),
        useMaterial3: true,
      ),
      home: const ChatScreen(),
    );
  }
}

// Status enum for feedback
enum FeedbackStatus { none, like, dislike }

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

  ChatMessage({
    required this.text,
    this.isUser = false,
    this.topic,
    this.sources = const [],
    this.isError = false,
    this.isSystemMessage = false,
    this.feedback = FeedbackStatus.none,
  }) : role = isUser ? 'user' : 'model';
}

class ChatScreen extends StatefulWidget {
  const ChatScreen({super.key});

  @override
  State<ChatScreen> createState() => _ChatScreenState();
}

class _ChatScreenState extends State<ChatScreen> {
  final TextEditingController _textController = TextEditingController();
  final ScrollController _scrollController = ScrollController();
  final List<ChatMessage> _messages = [];
  bool _isLoading = false;

  @override
  void initState() {
    super.initState();
    _addWelcomeMessage();
  }

  /// Adds the initial system welcome message
  void _addWelcomeMessage() {
    _messages.insert(
      0,
      ChatMessage(
        text: "<p>Hello! Ask me anything about the documents I have.</p>",
        isSystemMessage: true,
      ),
    );
  }

  @override
  void dispose() {
    _textController.dispose();
    _scrollController.dispose();
    super.dispose();
  }

  // ===========================================================================
  // CORE API LOGIC
  // ===========================================================================

  /// Polls the status endpoint until the task is completed or failed
  Future<Map<String, dynamic>> _pollForResponse(String taskId) async {
    final String statusUrl = "${AppSettings.apiUrl}/status/$taskId";

    // Safety 1: Set a maximum timeout (e.g., 60 seconds)
    final DateTime startTime = DateTime.now();
    const int timeoutSeconds = 60;

    // Safety 2: Track consecutive network errors to avoid crashing on a single blip
    int consecutiveErrors = 0;
    const int maxErrors = 3;

    while (true) {
      // Check Timeout
      if (DateTime.now().difference(startTime).inSeconds > timeoutSeconds) {
        throw Exception("Timeout: The request took too long to process.");
      }

      // 1. Wait before checking (Dynamic Backoff is cleaner, but 2s is fine)
      await Future.delayed(const Duration(seconds: 2));

      // 2. Check Status
      try {
        final response = await http.get(Uri.parse(statusUrl));

        // Reset error counter on successful connection
        consecutiveErrors = 0;

        if (response.statusCode == 200) {
          final data = jsonDecode(utf8.decode(response.bodyBytes));
          final String status = data['status'];

          if (status == 'completed' || status == 'success') {
            // SUCCESS
            return data['data'] ?? data;
          } else if (status == 'failed') {
            // WORKER FAILURE
            return {
              'status': 'failed',
              'error': data['error'] ?? "Worker error",
            };
          }
          // If 'processing' or 'PENDING', continue loop
        } else {
          // HTTP Server Error (500, 404, etc) - Count as an error
          consecutiveErrors++;
        }
      } catch (e) {
        consecutiveErrors++;
        // If we hit max errors, THEN crash. Otherwise, retry.
        if (consecutiveErrors >= maxErrors) {
          throw Exception("Connection lost. Unable to retrieve status.");
        }
      }
    }
  }

  /// Sends the user's query and chat history to the Flask backend
  Future<void> _handleSendPressed() async {
    final text = _textController.text;
    if (text.isEmpty) return;

    // 1. Add user message to UI immediately
    final userMessage = ChatMessage(text: text, isUser: true);
    _addMessage(userMessage);
    _textController.clear();

    // 2. Set Loading State
    setState(() {
      _isLoading = true;
    });

    // 3. Prepare Chat History
    // Filter out errors/system messages and format for backend
    final chatHistory = _messages
        .where((msg) => !msg.isError && !msg.isSystemMessage)
        .toList()
        .reversed // Oldest to newest
        .map((msg) => {'role': msg.role, 'text': msg.text})
        .toList();

    // Flask expects history without the current query
    final chatHistoryForApi = chatHistory.length > 1
        ? chatHistory.sublist(0, chatHistory.length - 1)
        : [];

    // The current query is in 'text' variable
    final lastQuery = text;

    try {
      final String chatUrl = "${AppSettings.apiUrl}/chat";

      // --- STEP A: Dispatch Task ---
      // We use a short timeout (10s) because the server should respond instantly with a Task ID
      final response = await http
          .post(
            Uri.parse(chatUrl),
            headers: {"Content-Type": "application/json"},
            body: jsonEncode({
              "history": chatHistoryForApi,
              "query": lastQuery,
            }),
          )
          .timeout(const Duration(seconds: 10));

      if (response.statusCode == 202) {
        // 202 Accepted = Task Started successfully
        final data = jsonDecode(utf8.decode(response.bodyBytes));
        final String taskId = data['task_id'];

        // --- STEP B: Poll for Result ---
        // This awaits until the loop in _pollForResponse finishes
        final resultData = await _pollForResponse(taskId);

        // --- STEP C: Handle Result ---
        // Check if the worker returned a logical error (like "Topic not found")
        // Your worker returns {"status": "not_found", ...} in these cases
        if (resultData['status'] == 'failed' ||
            resultData['status'] == 'not_found') {
          _addMessage(
            ChatMessage(
              text:
                  "<p><strong>Error:</strong> ${resultData['error'] ?? resultData['message']}</p>",
              isError: true,
              isSystemMessage: true,
            ),
          );
        } else {
          // Success!
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
        // The Dispatch failed immediately (e.g., 400 Bad Request, 500 Server Error)
        final data = jsonDecode(utf8.decode(response.bodyBytes));
        _addMessage(
          ChatMessage(
            text:
                "<p>Server Error (${response.statusCode}): ${data['message'] ?? 'Unknown error'}</p>",
            isError: true,
            isSystemMessage: true,
          ),
        );
      }
    } catch (e) {
      // Network Error or Polling Error
      _addMessage(
        ChatMessage(
          text: "<p>Connection failed. Error: ${e.toString()}</p>",
          isError: true,
          isSystemMessage: true,
        ),
      );
    } finally {
      // 4. Reset Loading State
      if (mounted) {
        setState(() {
          _isLoading = false;
        });
      }
    }
  }

  // --- Feedback Logic ---
  Future<void> _sendFeedback(ChatMessage message, bool isLike) async {
    // Prevent sending feedback for system/error messages or user messages
    if (message.isUser || message.isError || message.isSystemMessage) return;

    // 1. Identify the user query that prompted this answer
    int msgIndex = _messages.indexOf(message);
    String userQuery = "";

    // The user query should be immediately after the AI response in the reversed list
    if (msgIndex != -1 && msgIndex + 1 < _messages.length) {
      userQuery = _messages[msgIndex + 1].text;
    }

    // 2. Construct History *up to* this exchange
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

    // 3. Update UI State immediately
    setState(() {
      message.feedback = isLike ? FeedbackStatus.like : FeedbackStatus.dislike;
    });

    // 4. Send to Backend
    try {
      final String apiUrl = "${AppSettings.apiUrl}/feedback";
      await http.post(
        Uri.parse(apiUrl),
        headers: {"Content-Type": "application/json"},
        body: jsonEncode({
          "query": userQuery,
          "answer": message.text,
          "topic_id": message.topic,
          "rating": isLike ? 1 : -1,
          "history": contextHistory,
        }),
      );

      if (mounted) {
        ScaffoldMessenger.of(context).showSnackBar(
          const SnackBar(
            content: Text("Thank you for your feedback!"),
            duration: Duration(seconds: 1),
          ),
        );
      }
    } catch (e) {
      if (mounted) {
        ScaffoldMessenger.of(context).showSnackBar(
          const SnackBar(
            content: Text("Failed to send feedback."),
            duration: Duration(seconds: 1),
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
    // Animate to the (new) top of the reversed list
    _scrollController.animateTo(
      0.0,
      duration: const Duration(milliseconds: 300),
      curve: Curves.easeOut,
    );
  }

  /// --- Opens Nginx Document URLs ---
  Future<void> _launchUrl(String urlString) async {
    final Uri url = Uri.parse(urlString);
    if (!await launchUrl(url, mode: LaunchMode.externalApplication)) {
      // Show an error message if it fails to launch
      _addMessage(
        ChatMessage(
          text: "<p>Could not open the document: $urlString</p>",
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
      const SnackBar(
        content: Text("Chat history cleared."),
        duration: Duration(seconds: 2),
      ),
    );
  }

  // ===========================================================================
  // UI BUILDING
  // ===========================================================================

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        title: const Text("AI Document Chat"),
        actions: [
          // Clear Chat Button
          IconButton(
            icon: const Icon(Icons.delete_sweep_outlined),
            tooltip: "Clear Chat",
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
              reverse: true, // Makes the list start from the bottom
              padding: const EdgeInsets.all(16.0),
              itemCount: _messages.length,
              itemBuilder: (context, index) {
                return _buildMessageBubble(_messages[index]);
              },
            ),
          ),
          if (_isLoading)
            const Padding(
              padding: EdgeInsets.symmetric(horizontal: 16.0, vertical: 8.0),
              child: LinearProgressIndicator(),
            ),
          _buildTextInputArea(),
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
      // USER MESSAGE: Plain Text, Selectable
      messageContent = SelectableText(
        message.text,
        style: theme.textTheme.bodyLarge!.copyWith(
          color: theme.colorScheme.onPrimary,
        ),
      );
    } else {
      // AI MESSAGE: HTML, Selectable Area
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

            // Feedback Buttons (Only for valid AI answers)
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
                      onPressed: () => _sendFeedback(message, true),
                      tooltip: "Good response",
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
                      onPressed: () => _sendFeedback(message, false),
                      tooltip: "Bad response",
                    ),
                  ],
                ),
              ),
          ],
        ),
      );
    }

    return Container(
      margin: const EdgeInsets.symmetric(vertical: 8.0),
      child: Row(
        mainAxisAlignment: isUser
            ? MainAxisAlignment.end
            : MainAxisAlignment.start,
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          // Avatar
          if (!isUser)
            CircleAvatar(
              backgroundColor: theme.colorScheme.secondary,
              child: Icon(
                message.isError
                    ? Icons.error_outline
                    : (message.isSystemMessage
                          ? Icons
                                .info_outline // Different icon for system
                          : Icons.computer),
                color: theme.colorScheme.onSecondary,
              ),
            ),
          if (isUser) const SizedBox(width: 40), // Spacer
          // Bubble Container
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
                  bottomLeft: isUser ? const Radius.circular(20) : Radius.zero,
                  bottomRight: isUser ? Radius.zero : const Radius.circular(20),
                ),
              ),
              // The content (SelectableText or SelectionArea)
              child: messageContent,
            ),
          ),

          if (isUser)
            CircleAvatar(
              backgroundColor: theme.colorScheme.primary,
              child: Icon(Icons.person, color: theme.colorScheme.onPrimary),
            ),
          if (!isUser) const SizedBox(width: 40), // Spacer
        ],
      ),
    );
  }

  /// Builds the source chips for an AI message
  Widget _buildSources(ChatMessage message) {
    if (message.isUser || message.sources.isEmpty) {
      return const SizedBox.shrink();
    }

    final theme = Theme.of(context);
    final String topic = message.topic ?? "unknown_topic";

    return Padding(
      padding: const EdgeInsets.only(top: 12.0),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          // Title text (Selectable due to parent SelectionArea)
          Text(
            "Sources (from topic: $topic):",
            style: theme.textTheme.bodySmall!.copyWith(
              fontWeight: FontWeight.bold,
              color: theme.colorScheme.onSurface.withAlpha(204),
            ),
          ),
          const SizedBox(height: 8.0),

          // Chips
          Wrap(
            spacing: 8.0,
            runSpacing: 4.0,
            children: message.sources.map((source) {
              final String fileName = source['file'] ?? 'Unknown File';

              // Construct the Nginx URL
              final String url =
                  "${AppSettings.downloadDocumentUrl}/$topic/$fileName";

              // Wrapper to enable cursor change on hover
              return MouseRegion(
                cursor: SystemMouseCursors.click,
                child: GestureDetector(
                  // Use GestureDetector or InkWell
                  onTap: () => _launchUrl(url),
                  child: Chip(
                    mouseCursor: SystemMouseCursors
                        .click, // 2. Set cursor directly on the Chip
                    avatar: Icon(
                      Icons.link,
                      size: 16,
                      color: theme.colorScheme.secondary,
                    ),
                    label: Text(fileName),
                    labelStyle: theme.textTheme.labelSmall,
                    backgroundColor: theme.colorScheme.secondaryContainer,
                    labelPadding: const EdgeInsets.symmetric(horizontal: 8.0),
                    visualDensity: VisualDensity.compact,
                    // Optional: adds a slight "clickable" feel
                    side: BorderSide(color: theme.colorScheme.outlineVariant),
                  ),
                ),
              );
            }).toList(),
          ),
        ],
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
            color: Colors.black.withOpacity(0.1),
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
                enabled: !_isLoading,
                autocorrect: true,
                enableSuggestions: true,
                decoration: InputDecoration(
                  hintText: "Send a message...",
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
              onPressed: _isLoading ? null : _handleSendPressed,
              backgroundColor: theme.colorScheme.primary,
              child: Icon(Icons.send, color: theme.colorScheme.onPrimary),
            ),
          ],
        ),
      ),
    );
  }
}

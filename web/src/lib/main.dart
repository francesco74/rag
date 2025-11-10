/*
 * ====================================================================
 * RAG AI Chat App (Flutter Client - Web Only)
 * ====================================================================
 *
 * This is the complete, final version of the app.
 *
 * It uses:
 * - A `SelectableText` widget for your (user) messages.
 * - An `Html` widget (in a SelectionArea) for the AI's HTML responses.
 *
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

// Simple data model for a chat message
class ChatMessage {
  final String text;
  final bool isUser;
  final String? topic;
  final List<Map<String, dynamic>> sources;
  final bool isError;
  final bool isSystemMessage; // <-- NEW: Flag to exclude from history
  // Added to support chat history
  final String role; // 'user' or 'model'

  ChatMessage({
    required this.text,
    this.isUser = false,
    this.topic,
    this.sources = const [],
    this.isError = false,
    this.isSystemMessage = false, // <-- NEW: Default to false
  }) : role = isUser ? 'user' : 'model'; // Automatically assign role
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
        isSystemMessage: true, // <-- NEW: Mark as system message
      ),
    );
  }

  @override
  void dispose() {
    _textController.dispose();
    _scrollController.dispose();
    super.dispose();
  }

  // --- Core API Logic ---

  /// Sends the user's query and chat history to the Flask backend
  Future<void> _handleSendPressed() async {
    final text = _textController.text;
    if (text.isEmpty) return;

    // Add user message to UI (as plain text)
    final userMessage = ChatMessage(text: text, isUser: true);
    _addMessage(userMessage);
    _textController.clear();

    // Show loading indicator
    setState(() {
      _isLoading = true;
    });

    // --- Prepare Chat History ---
    // Create a list of all non-error AND non-system messages
    final chatHistory = _messages
        .where((msg) =>
            !msg.isError && !msg.isSystemMessage) // <-- UPDATED FILTER
        .toList() // Convert Iterable to List
        .reversed // Oldest to newest
        .map((msg) => {
              'role': msg.role,
              'text': msg.text,
            })
        .toList(); // Convert back to List

    // The Flask server now expects the *last* query separately
    final chatHistoryForApi = chatHistory.length > 1
        ? chatHistory.sublist(0, chatHistory.length - 1)
        : [];
    final lastQuery = chatHistory.last['text'] ?? "";

    // Call the API
    try {
      final String apiUrl = "${AppSettings.apiUrl}/chat";

      final response = await http
          .post(
            Uri.parse(apiUrl),
            headers: {"Content-Type": "application/json"},
            // Send the history and the last query
            body: jsonEncode({
              "history": chatHistoryForApi, // Send all *but* the last message
              "query": lastQuery // Send the last query
            }),
          )
          .timeout(const Duration(seconds: 90));

      // Handle the response
      if (response.statusCode == 200) {
        final data = jsonDecode(utf8.decode(response.bodyBytes));
        _addMessage(ChatMessage(
          text: data['answer'],
          topic: data['topic'],
          // Ensure sources is correctly casted
          sources: (data['sources'] as List)
              .map((s) => s as Map<String, dynamic>)
              .toList(),
        ));
      } else if (response.statusCode == 404) {
        // Handle "Not Found" errors (no topic, no answer)
        final data = jsonDecode(utf8.decode(response.bodyBytes));
        _addMessage(ChatMessage(
          text: data['message'] ?? "<p>I could not find an answer.</p>",
          isError: true,
          isSystemMessage: true, // <-- NEW: Mark as system message
        ));
      } else {
        // Handle other server errors (500, etc.)
        final data = jsonDecode(utf8.decode(response.bodyBytes));
        _addMessage(ChatMessage(
          text:
              "<p>An error occurred on the server (Code: ${response.statusCode}).</p><p>${data['message'] ?? ''}</p>",
          isError: true,
          isSystemMessage: true, // <-- NEW: Mark as system message
        ));
      }
    } catch (e) {
      // Handle network/connection errors
      _addMessage(ChatMessage(
        text:
            "<p>Failed to connect to the AI engine. Is the server running at ${AppSettings.apiUrl}?</p><p>Error: ${e.toString()}</p>",
        isError: true,
        isSystemMessage: true, // <-- NEW: Mark as system message
      ));
    } finally {
      // Hide loading indicator
      setState(() {
        _isLoading = false;
      });
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

  /// --- New Function to Launch URLs ---
  Future<void> _launchUrl(String urlString) async {
    final Uri url = Uri.parse(urlString);
    if (!await launchUrl(
      url,
      mode: LaunchMode.externalApplication,
    )) {
      // Show an error message if it fails to launch
      _addMessage(ChatMessage(
        text: "<p>Could not open the document: $urlString</p>",
        isError: true,
        isSystemMessage: true, // <-- NEW: Mark as system message
      ));
    }
  }

  // --- NEW: Function to clear the chat ---
  void _clearChat() {
    setState(() {
      _messages.clear();
      _addWelcomeMessage();
    });

    // Show a confirmation snackbar
    ScaffoldMessenger.of(context).showSnackBar(
      const SnackBar(
        content: Text("Chat history cleared."),
        duration: Duration(seconds: 2),
      ),
    );
  }

  // --- UI Building Widgets ---

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        title: const Text("AI Document Chat"),
        // --- NEW: Clear chat button ---
        actions: [
          IconButton(
            icon: const Icon(Icons.delete_sweep_outlined),
            tooltip: "Clear Chat",
            onPressed: _isLoading ? null : _clearChat,
          ),
          const SizedBox(width: 8),
        ],
        // --- END NEW ---
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

    // --- THIS IS THE FIX ---
    // Use a SelectableText widget for user messages (plain text)
    // Use an Html widget (in a SelectionArea) for AI/error messages (HTML)
    Widget messageContent;
    if (isUser) {
      // USER MESSAGE
      messageContent = SelectableText(
        message.text,
        style: theme.textTheme.bodyLarge!
            .copyWith(color: theme.colorScheme.onPrimary),
      );
    } else {
      // AI MESSAGE (or error)
      messageContent = SelectionArea(
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            Html(
              data: message.text,
              style: {
                "body": Style(
                  margin: Margins.zero,
                  padding: HtmlPaddings.zero,
                ),
                "p": Style(
                  fontSize: FontSize(
                      Theme.of(context).textTheme.bodyLarge!.fontSize!),
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
          ],
        ),
      );
    }
    // --- END FIX ---

    return Container(
      margin: const EdgeInsets.symmetric(vertical: 8.0),
      child: Row(
        mainAxisAlignment:
            isUser ? MainAxisAlignment.end : MainAxisAlignment.start,
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
                        ? Icons.info_outline // Different icon for system
                        : Icons.computer),
                color: theme.colorScheme.onSecondary,
              ),
            ),
          if (isUser) const SizedBox(width: 40), // Spacer

          // Bubble
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
              // We now apply selection *inside* the bubble
              child: messageContent,
            ),
          ),

          if (isUser)
            CircleAvatar(
              backgroundColor: theme.colorScheme.primary,
              child: Icon(
                Icons.person,
                color: theme.colorScheme.onPrimary,
              ),
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
          // This text is now selectable because it's inside the SelectionArea
          Text(
            "Sources (from topic: $topic):",
            style: theme.textTheme.bodySmall!.copyWith(
              fontWeight: FontWeight.bold,
              color: theme.colorScheme.onSurface.withAlpha(204),
            ),
          ),
          const SizedBox(height: 8.0),
          Wrap(
            spacing: 8.0,
            runSpacing: 4.0,
            children: message.sources.map((source) {
              final String fileName = source['file'] ?? 'Unknown File';
              // --- "Whole Document" Change ---
              // Page number is no longer reliable, so we don't show it.
              // final String page = (source['page'] ?? 0).toString();

              // Construct the Nginx URL
              final String url =
                  "${AppSettings.downloadDocumentUrl}/$topic/$fileName";

              return InkWell(
                onTap: () => _launchUrl(url),
                borderRadius: BorderRadius.circular(16.0),
                child: Chip(
                  avatar: Icon(Icons.link,
                      size: 16, color: theme.colorScheme.secondary),
                  // --- "Whole Document" Change ---
                  label: Text(fileName), // No page number
                  labelStyle: theme.textTheme.labelSmall,
                  backgroundColor: theme.colorScheme.secondaryContainer,
                  labelPadding: const EdgeInsets.symmetric(horizontal: 8.0),
                  visualDensity: VisualDensity.compact,
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
    final theme = Theme.of(context); // Get theme here
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
              child: Icon(
                Icons.send,
                color: theme.colorScheme.onPrimary,
              ),
            ),
          ],
        ),
      ),
    );
  }
}
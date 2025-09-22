from PyQt5.QtWidgets import QTextEdit, QListWidget, QListWidgetItem, QLabel
from PyQt5.QtCore import Qt, QEvent, pyqtSignal
from PyQt5.QtGui import QFontMetrics, QColor, QTextCursor, QTextCharFormat
from ..styles import TEXT_EDIT_STYLE, LIST_STYLE, NEXT_TOKEN_OVERLAY_STYLE


class AutocompleteTextEdit(QTextEdit):
    """Text input with intelligent autocomplete and validation"""

    # Signals
    text_validated = pyqtSignal(str)
    prediction_requested = pyqtSignal(str)

    def __init__(self, vocab_manager, model_manager):
        super().__init__()
        self.vocab_manager = vocab_manager
        self.model_manager = model_manager

        # Autocomplete variables
        self.current_suggestions = []
        self.suggestion_index = -1
        self.current_token_start = 0
        self.current_token_end = 0
        self.is_validating = False

        # Setup UI
        self.setPlaceholderText("Enter a sentence to analyse...")
        self.setStyleSheet(TEXT_EDIT_STYLE)
        self.installEventFilter(self)

        # Semi-transparent next token suggestion overlay
        self.next_token_overlay = QLabel("")
        self.next_token_overlay.setStyleSheet(NEXT_TOKEN_OVERLAY_STYLE)
        self.next_token_overlay.setVisible(False)
        self.next_token_overlay.setWindowFlags(Qt.WindowType.FramelessWindowHint | Qt.WindowType.Tool)
        self.next_token_overlay.setAttribute(Qt.WA_TranslucentBackground)

        # Autocomplete suggestions list (overlay)
        self.suggestions_list = QListWidget()
        self.suggestions_list.setMaximumHeight(120)
        self.suggestions_list.setVisible(False)
        self.suggestions_list.itemClicked.connect(self.on_suggestion_selected)
        self.suggestions_list.setStyleSheet(LIST_STYLE)
        self.suggestions_list.setWindowFlags(Qt.WindowType.FramelessWindowHint | Qt.WindowType.Tool)
        self.suggestions_list.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.suggestions_list.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)

        # Connect text change signal
        self.textChanged.connect(self.on_text_changed)

    def eventFilter(self, obj, event):
        if obj == self:
            if event.type() == QEvent.Type.KeyPress:
                return self.handle_key_press(event)
            elif event.type() == QEvent.Type.FocusOut:
                self.hide_suggestions()
        return super().eventFilter(obj, event)

    def handle_key_press(self, event):
        key = event.key()

        if key == Qt.Key.Key_Down:
            self.navigate_suggestions(1)
            return True
        elif key == Qt.Key.Key_Up:
            self.navigate_suggestions(-1)
            return True
        elif key == Qt.Key.Key_Escape:
            self.hide_suggestions()
            return True
        elif key == Qt.Key.Key_Space:
            self.hide_suggestions()
            self.show_next_token_suggestion()
            return False
        elif key == Qt.Key.Key_Period:
            if self.suggestions_list.isVisible() and self.current_suggestions:
                current_token = self._get_current_token()
                if current_token and any(
                    suggestion.lower() == current_token.lower() for suggestion in self.current_suggestions
                ):
                    exact_match = next(
                        suggestion
                        for suggestion in self.current_suggestions
                        if suggestion.lower() == current_token.lower()
                    )
                    self._accept_exact_match_with_period(exact_match)
                    return True
            return False
        elif key == Qt.Key.Key_Tab:
            if self.next_token_overlay.isVisible():
                self.accept_next_token_suggestion()
                return True
            elif self.suggestions_list.isVisible() and self.suggestion_index >= 0:
                self.select_current_suggestion(insert_space=True)
                return True
            else:
                return True
        elif key == Qt.Key.Key_Backspace:
            self.update_suggestions()
            return False
        else:
            self.update_suggestions()
            return False

        return False

    def update_suggestions(self):
        cursor = self.textCursor()
        text = self.toPlainText()
        cursor_pos = cursor.position()

        # Find the current token being typed
        token_start = cursor_pos
        token_end = cursor_pos

        # Find token boundaries
        while token_start > 0 and text[token_start - 1].isalnum():
            token_start -= 1
        while token_end < len(text) and text[token_end].isalnum():
            token_end += 1

        current_token = text[token_start:token_end].lower()

        if len(current_token) < 1:
            self.hide_suggestions()
            # Show next token suggestion if we have valid tokens
            text = self.toPlainText().strip()
            if text:
                tokens = text.split()
                if tokens and self.vocab_manager.is_token_valid(tokens[-1].lower()):
                    self.show_next_token_suggestion()
                else:
                    self.next_token_overlay.setVisible(False)
            else:
                self.next_token_overlay.setVisible(False)
            return

        # Check if current token is valid
        if self.vocab_manager.is_token_valid(current_token):
            self.hide_suggestions()
            self.show_next_token_suggestion()
            return

        # Filter vocabulary for suggestions
        all_matches = self.vocab_manager.get_token_suggestions(current_token)

        if not all_matches:
            self.hide_suggestions()
            return

        # Separate exact matches from partial matches
        exact_matches = [token for token in all_matches if token.lower() == current_token]
        partial_matches = [token for token in all_matches if token.lower() != current_token]

        # Prioritise exact matches first, then partial matches alphabetically
        exact_matches.sort()
        partial_matches.sort()
        suggestions = exact_matches + partial_matches

        # Update suggestions list
        self.current_suggestions = suggestions[:10]
        self.suggestions_list.clear()

        for i, suggestion in enumerate(self.current_suggestions):
            item = QListWidgetItem(suggestion)
            if suggestion.lower() == current_token:
                item.setBackground(QColor("#2d5a2d"))
                item.setForeground(QColor("#90EE90"))
            self.suggestions_list.addItem(item)

        self.current_token_start = token_start
        self.current_token_end = token_end
        self.suggestion_index = 0
        self.update_suggestion_selection()

        # Position and show suggestions list
        cursor = self.textCursor()
        cursor.setPosition(self.current_token_start)
        rect = self.cursorRect(cursor)
        pos = self.viewport().mapToGlobal(rect.bottomLeft())
        self.suggestions_list.move(pos)

        # Calculate width
        if self.current_suggestions:
            metrics = QFontMetrics(self.font())
            max_width = max(metrics.width(token) for token in self.current_suggestions)
            max_width += 40
            max_width = min(max_width, self.width())
        else:
            max_width = 100

        # Adjust position
        if pos.x() + max_width > self.parent().width():
            pos.setX(self.parent().width() - max_width)
        if pos.x() < 0:
            pos.setX(0)

        # Calculate height
        if self.suggestions_list.count() > 0:
            last_item_rect = self.suggestions_list.visualItemRect(
                self.suggestions_list.item(self.suggestions_list.count() - 1)
            )
            total_height = last_item_rect.bottom() + self.suggestions_list.contentsMargins().bottom()
        else:
            total_height = 120

        self.suggestions_list.resize(max_width, min(120, total_height))
        self.suggestions_list.setVisible(True)
        self.suggestions_list.setCurrentRow(0)
        self.suggestions_list.raise_()

    def navigate_suggestions(self, direction):
        if not self.suggestions_list.isVisible():
            return

        self.suggestion_index += direction
        if self.suggestion_index < 0:
            self.suggestion_index = len(self.current_suggestions) - 1
        elif self.suggestion_index >= len(self.current_suggestions):
            self.suggestion_index = 0

        self.update_suggestion_selection()

    def update_suggestion_selection(self):
        if self.suggestion_index >= 0 and self.suggestion_index < len(self.current_suggestions):
            self.suggestions_list.setCurrentRow(self.suggestion_index)

    def select_current_suggestion(self, insert_space=False):
        if not self.suggestions_list.isVisible() or self.suggestion_index < 0:
            return

        selected_token = self.current_suggestions[self.suggestion_index]
        cursor = self.textCursor()
        text = self.toPlainText()

        if (
            self.current_token_start == self.current_token_end == len(text)
            and self.current_token_start > 0
            and text[self.current_token_start - 1] != " "
        ):
            cursor.setPosition(self.current_token_start)
            cursor.insertText(" ")
        cursor.setPosition(self.current_token_start)
        cursor.setPosition(self.current_token_end, QTextCursor.KeepAnchor)
        cursor.insertText(selected_token)
        if insert_space:
            cursor.insertText(" ")

        self.hide_suggestions()

    def on_suggestion_selected(self, item):
        selected_token = item.text()
        cursor = self.textCursor()
        cursor.setPosition(self.current_token_start)
        cursor.setPosition(self.current_token_end, QTextCursor.KeepAnchor)
        cursor.insertText(selected_token)

        self.hide_suggestions()

    def hide_suggestions(self):
        self.suggestions_list.setVisible(False)
        self.current_suggestions = []
        self.suggestion_index = -1

    def hide_all_overlays(self):
        """Hide all autocomplete overlays (suggestions list and next token overlay)"""
        self.hide_suggestions()
        self.next_token_overlay.setVisible(False)

    def show_next_token_suggestion(self):
        text = self.toPlainText().strip()
        if not text:
            self.next_token_overlay.setVisible(False)
            return

        tokens = text.split()
        if not tokens:
            self.next_token_overlay.setVisible(False)
            return

        # Check if the last token is invalid - don't show suggestions for invalid tokens
        last_token = tokens[-1].lower()
        if not self.vocab_manager.is_token_valid(last_token):
            self.next_token_overlay.setVisible(False)
            return

        # Get prediction from model
        predictions = self.model_manager.predict_next_tokens(text, top_k=1)
        if not predictions:
            self.next_token_overlay.setVisible(False)
            return

        next_token = predictions[0][0]

        # Check if there's already a space at cursor position
        cursor = self.textCursor()
        text = self.toPlainText()
        pos = cursor.position()
        needs_space = pos == 0 or text[pos - 1] != " "

        if needs_space:
            self.next_token_overlay.setText(" " + next_token)
        else:
            self.next_token_overlay.setText(next_token)

        self.next_token_overlay.setVisible(True)
        rect = self.cursorRect(cursor)
        pos = self.viewport().mapToGlobal(rect.topLeft())
        self.next_token_overlay.move(pos)
        self.next_token_overlay.adjustSize()
        self.next_token_overlay.raise_()

    def accept_next_token_suggestion(self):
        if self.next_token_overlay.isVisible():
            suggestion = self.next_token_overlay.text().lstrip()
            cursor = self.textCursor()
            text = self.toPlainText()
            pos = cursor.position()
            if pos > 0 and text[pos - 1] != " ":
                cursor.insertText(" ")
            cursor.insertText(suggestion + " ")
            self.next_token_overlay.setVisible(False)
            self.hide_suggestions()
            self.update_suggestions()

    def on_text_changed(self):
        self.validate_and_highlight_text()
        self.update_suggestions()
        # Only show next token suggestion if there's text and it's valid
        text = self.toPlainText().strip()
        if text:
            tokens = text.split()
            if tokens and self.vocab_manager.is_token_valid(tokens[-1].lower()):
                self.show_next_token_suggestion()
            else:
                self.next_token_overlay.setVisible(False)
        else:
            self.next_token_overlay.setVisible(False)
        # Emit signal to update predictions in results section
        if text:
            self.prediction_requested.emit(text)

    def validate_and_highlight_text(self):
        if self.is_validating:
            return

        self.is_validating = True
        try:
            text = self.toPlainText()
            if not text.strip():
                return

            cursor = QTextCursor(self.document())
            cursor.select(QTextCursor.Document)

            default_format = QTextCharFormat()
            default_format.setForeground(QColor("white"))

            invalid_format = QTextCharFormat()
            invalid_format.setForeground(QColor("red"))

            cursor.setCharFormat(default_format)
            cursor.clearSelection()

            import re

            for match in re.finditer(r"\b\w+\b", text):
                token = match.group().lower()
                start = match.start()
                end = match.end()

                should_validate = False
                if end == len(text):
                    should_validate = False
                elif end < len(text):
                    next_char = text[end]
                    if next_char in " \t\n,.!?":
                        should_validate = True

                if should_validate:
                    is_valid = self.vocab_manager.is_token_valid(token)
                    if not is_valid:
                        cursor.setPosition(start)
                        cursor.setPosition(end, QTextCursor.KeepAnchor)
                        cursor.setCharFormat(invalid_format)

            cursor = self.textCursor()
            cursor.setPosition(len(text))
            self.setTextCursor(cursor)
        finally:
            self.is_validating = False

    def _get_current_token(self):
        cursor = self.textCursor()
        text = self.toPlainText()
        cursor_pos = cursor.position()
        token_start = cursor_pos
        token_end = cursor_pos

        while token_start > 0 and text[token_start - 1].isalnum():
            token_start -= 1
        while token_end < len(text) and text[token_end].isalnum():
            token_end += 1

        return text[token_start:token_end].lower()

    def _accept_exact_match_with_period(self, exact_match):
        cursor = self.textCursor()
        cursor.setPosition(self.current_token_start)
        cursor.setPosition(self.current_token_end, QTextCursor.KeepAnchor)
        cursor.insertText(exact_match + ".")

        self.hide_suggestions()

# Style definitions

GROUP_STYLE = """
    QGroupBox {
        font-size: 14px;
        font-weight: bold;
        border: 2px solid #4c4c4c;
        border-radius: 8px;
        margin-top: 10px;
        padding-top: 10px;
        background-color: #2b2b2b;
        color: #c0c0c0;
    }
    QGroupBox::title {
        subcontrol-origin: margin;
        left: 10px;
        padding: 0 5px 0 5px;
        color: #c0c0c0;
        font-size: 16px;
    }
"""

BUTTON_STYLE = """
    QPushButton {
        background-color: #4c4c4c;
        color: #c0c0c0;
        border: none;
        padding: 12px 24px;
        border-radius: 6px;
        font-size: 14px;
        font-weight: bold;
    }
    QPushButton:hover {
        background-color: #5c5c5c;
    }
    QPushButton:pressed {
        background-color: #3c3c3c;
    }
    QPushButton:disabled {
        background-color: #2c2c2c;
        color: #666666;
    }
"""

INPUT_STYLE = """
    QLineEdit {
        background-color: #1e1e1e;
        color: #c0c0c0;
        border: 1px solid #4c4c4c;
        border-radius: 4px;
        padding: 8px;
        font-size: 14px;
    }
    QLineEdit:focus {
        border: 1px solid #6c6c6c;
    }
"""

SLIDER_STYLE = """
    QSlider::groove:horizontal {
        background: #3c3c3c;
        height: 8px;
        border-radius: 4px;
    }
    QSlider::handle:horizontal {
        background: #6c6c6c;
        width: 20px;
        height: 20px;
        border-radius: 10px;
        margin: -6px 0;
    }
    QSlider::handle:horizontal:hover {
        background: #8c8c8c;
    }
"""

PROGRESS_STYLE = """
    QProgressBar {
        border: 1px solid #4c4c4c;
        border-radius: 4px;
        text-align: center;
        background-color: #1e1e1e;
        color: #c0c0c0;
        font-size: 12px;
    }
    QProgressBar::chunk {
        background-color: #4caf50;
        border-radius: 3px;
    }
"""

COMBO_STYLE = """
    QComboBox {
        background-color: #2b2b2b;
        color: #c0c0c0;
        border: 1px solid #4c4c4c;
        border-radius: 6px;
        padding: 10px 12px;
        font-size: 14px;
        min-width: 200px;
    }
    QComboBox:hover {
        border: 1px solid #6c6c6c;
    }
    QComboBox:focus {
        border: 1px solid #8c8c8c;
    }
    QComboBox::drop-down {
        border: none;
        width: 30px;
    }
    QComboBox::down-arrow {
        image: none;
        border-left: 5px solid transparent;
        border-right: 5px solid transparent;
        border-top: 5px solid #c0c0c0;
        margin-right: 10px;
    }
    QComboBox::down-arrow:hover {
        border-top: 5px solid #ffffff;
    }
    QComboBox QAbstractItemView {
        background-color: #2b2b2b;
        color: #c0c0c0;
        border: 1px solid #4c4c4c;
        border-radius: 6px;
        selection-background-color: #4c4c4c;
        selection-color: #ffffff;
        outline: none;
        padding: 4px;
    }
    QComboBox QAbstractItemView::item {
        padding: 16px 20px;
        border-radius: 4px;
        margin: 6px;
        min-height: 24px;
    }
    QComboBox QAbstractItemView::item:hover {
        background-color: #3c3c3c;
    }
    QComboBox QAbstractItemView::item:selected {
        background-color: #4c4c4c;
        color: #ffffff;
    }
"""

LIST_STYLE = """
    QListWidget {
        background-color: #1e1e1e;
        color: #c0c0c0;
        border: 1px solid #4c4c4c;
        border-radius: 4px;
        font-size: 12px;
    }
    QListWidget::item {
        padding: 5px;
        border-bottom: 1px solid #3c3c3c;
    }
    QListWidget::item:selected {
        background-color: #4c4c4c;
    }
"""

INPUT_GROUP_STYLE = """
    QGroupBox {
        font-size: 14px;
        font-weight: bold;
        border: 2px solid #4c4c4c;
        border-radius: 8px;
        margin-top: 5px;
        padding-top: 2px;
        background-color: #2b2b2b;
        color: #c0c0c0;
    }
    QGroupBox::title {
        subcontrol-origin: margin;
        left: 10px;
        padding: 0 5px 0 5px;
        color: #c0c0c0;
        font-size: 16px;
    }
"""

TEXT_EDIT_STYLE = """
    QTextEdit {
        background-color: #2b2b2b;
        color: #c0c0c0;
        border: 1px solid #4c4c4c;
        border-radius: 4px;
        padding: 8px;
        font-size: 14px;
    }
    QTextEdit:focus {
        border: 1px solid #6c6c6c;
    }
    QTextEdit::placeholder {
        color: #888888;
    }
"""

NEXT_TOKEN_OVERLAY_STYLE = """
    QLabel {
        color: rgba(192, 192, 192, 128);
        background-color: transparent;
        font-size: 14px;
        padding: 0px;
        margin: 0px;
    }
"""

STOP_BUTTON_STYLE = """
    QPushButton {
        background-color: #8B0000;
        color: #c0c0c0;
        border: none;
        padding: 12px 24px;
        border-radius: 6px;
        font-size: 14px;
        font-weight: bold;
    }
    QPushButton:hover {
        background-color: #A00000;
    }
    QPushButton:pressed {
        background-color: #700000;
    }
    QPushButton:disabled {
        background-color: #2c2c2c;
        color: #666666;
    }
"""

MAIN_WINDOW_STYLE = """
    QMainWindow {
        background-color: #1a1a1a;
        color: #c0c0c0;
    }
    QLabel {
        color: #c0c0c0;
    }
    QTabWidget {
        background-color: #1a1a1a;
    }
    QTabBar {
        background-color: #1a1a1a;
    }
    QTabWidget::pane {
        border: 1px solid #3c3c3c;
        background-color: #2b2b2b;
    }
    QTabBar::tab {
        background-color: #3c3c3c;
        color: #c0c0c0;
        padding: 10px 20px;
        margin-right: 2px;
        border-radius: 5px 5px 0 0;
    }
    QTabBar::tab:selected {
        background-color: #4c4c4c;
        color: #c0c0c0;
    }
    QTabBar::tab:hover {
        background-color: #4c4c4c;
    }
"""

RESULTS_LABEL_STYLE = "font-size: 14px; padding: 10px; background-color: #1e1e1e; border-radius: 5px; color: #c0c0c0;"

TRAINING_STATUS_STYLE = "font-size: 14px; padding: 10px; background-color: #1e1e1e; border-radius: 5px; color: #c0c0c0;"

TIME_REMAINING_STYLE = "font-size: 14px; padding: 10px; background-color: #1e1e1e; border-radius: 5px; color: #c0c0c0;"

INFO_LABEL_STYLE = "color: #c0c0c0;"

INFO_BUTTON_STYLE = "color: #007bff; font-size: 12px; font-weight: bold;"

AVAILABLE_LABEL_STYLE = "color: #c0c0c0;"

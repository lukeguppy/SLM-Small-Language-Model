from .gui.main_gui import MainGUI
from PyQt5.QtWidgets import QApplication
from .gui.styles import MAIN_WINDOW_STYLE
from .core.logger import get_main_logger

if __name__ == "__main__":
    logger = get_main_logger()
    logger.info("Starting SLM application")

    app = QApplication([])
    app.setStyleSheet(MAIN_WINDOW_STYLE)

    window = MainGUI()
    window.show()
    
    app.exec_()

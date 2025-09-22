from PyQt5.QtWidgets import QProgressBar


class CustomProgressBar(QProgressBar):
    """Custom progress bar with percentage display"""

    def text(self):
        """Override to show percentage with 2 decimal places"""
        percentage = self.value() / self.maximum() * 100
        return "%.2f%%" % percentage

"""
Steganography Comparison Desktop Application

A PyQt6 desktop app for comparing 4 steganography embedding methods
(LSB Random Spatial, LSB Canny-Sobel, PVD Sequential, LSBMR)
on paired natural/AI-generated images.

Usage:
    python app.py
"""

import sys
from pathlib import Path

from PyQt6.QtWidgets import QApplication

from gui.main_window import MainWindow


def main():
    """Launch the application."""
    app = QApplication(sys.argv)
    
    # Set application properties
    app.setApplicationName("Steganography Comparison")
    app.setApplicationVersion("1.0")
    
    # Create and show main window
    window = MainWindow()
    window.show()
    
    # Run application
    sys.exit(app.exec())


if __name__ == "__main__":
    main()

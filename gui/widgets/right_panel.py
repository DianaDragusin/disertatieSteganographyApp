"""Main application window right panel."""
from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QLabel, QTabWidget, QTextEdit
)
from PyQt6.QtGui import QFont

from gui.widgets.metrics_table import MetricsTableWidget


class RightPanel(QWidget):
    """Right results panel with tabbed metrics and extraction."""
    
    def __init__(self, parent=None):
        """Initialize right panel."""
        super().__init__(parent)
        self.init_ui()
    
    def init_ui(self):
        """Initialize UI."""
        layout = QVBoxLayout()
        
        tabs = QTabWidget()
        
        # Metrics tab
        self.metrics_table = MetricsTableWidget()
        tabs.addTab(self.metrics_table, "Metrics")
        
        # Extraction tab
        extract_widget = QWidget()
        extract_layout = QVBoxLayout(extract_widget)
        
        extract_title = QLabel("Extraction Results")
        extract_title.setFont(QFont("Arial", 11, QFont.Weight.Bold))
        extract_layout.addWidget(extract_title)
        
        extract_result_label = QLabel("Extracted from Natural:")
        extract_layout.addWidget(extract_result_label)
        
        self.extract_natural_text = QTextEdit()
        self.extract_natural_text.setReadOnly(True)
        self.extract_natural_text.setFixedHeight(60)
        extract_layout.addWidget(self.extract_natural_text)
        
        self.match_natural_label = QLabel("Match: ✗")
        self.match_natural_label.setStyleSheet("color: red; font-weight: bold;")
        extract_layout.addWidget(self.match_natural_label)
        
        extract_result_label2 = QLabel("Extracted from AI:")
        extract_layout.addWidget(extract_result_label2)
        
        self.extract_ai_text = QTextEdit()
        self.extract_ai_text.setReadOnly(True)
        self.extract_ai_text.setFixedHeight(60)
        extract_layout.addWidget(self.extract_ai_text)
        
        self.match_ai_label = QLabel("Match: ✗")
        self.match_ai_label.setStyleSheet("color: red; font-weight: bold;")
        extract_layout.addWidget(self.match_ai_label)
        
        extract_layout.addStretch()
        tabs.addTab(extract_widget, "Extraction")
        
        layout.addWidget(tabs)
        self.setLayout(layout)
    
    def update_extraction(self, result: dict):
        """Update extraction results."""
        self.extract_natural_text.setText(result['natural'])
        self.extract_ai_text.setText(result['ai'])
        
        if result['match_natural']:
            self.match_natural_label.setText("Match: ✓")
            self.match_natural_label.setStyleSheet("color: green; font-weight: bold;")
        else:
            self.match_natural_label.setText("Match: ✗")
            self.match_natural_label.setStyleSheet("color: red; font-weight: bold;")
        
        if result['match_ai']:
            self.match_ai_label.setText("Match: ✓")
            self.match_ai_label.setStyleSheet("color: green; font-weight: bold;")
        else:
            self.match_ai_label.setText("Match: ✗")
            self.match_ai_label.setStyleSheet("color: red; font-weight: bold;")

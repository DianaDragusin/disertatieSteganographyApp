"""Metrics display widget showing AI vs Natural comparison."""
from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QTableWidget,
    QTableWidgetItem, QHeaderView
)
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QColor, QFont

from core.models import ComparisonResult


class MetricsTableWidget(QWidget):
    """Side-by-side metrics display for AI vs Natural."""
    
    def __init__(self, parent=None):
        """Initialize metrics table."""
        super().__init__(parent)
        self.comparison_result: ComparisonResult = None
        
        self.init_ui()
    
    def init_ui(self):
        """Initialize UI layout."""
        layout = QVBoxLayout()
        
        # Title
        title = QLabel("Comparison Results")
        title_font = QFont()
        title_font.setPointSize(14)
        title_font.setBold(True)
        title.setFont(title_font)
        layout.addWidget(title)
        
        # Create table with 3 columns: Metric, AI, Natural
        self.table = QTableWidget()
        self.table.setColumnCount(3)
        self.table.setHorizontalHeaderLabels(["Metric", "AI Image", "Natural Image"])
        
        # Set column widths
        header = self.table.horizontalHeader()
        header.setSectionResizeMode(0, QHeaderView.ResizeMode.Stretch)
        header.setSectionResizeMode(1, QHeaderView.ResizeMode.Stretch)
        header.setSectionResizeMode(2, QHeaderView.ResizeMode.Stretch)
        
        # Set row count
        self.table.setRowCount(9)  # 8 metrics + 1 winner row
        
        # Populate metric labels
        metrics = [
            "PSNR (dB)",
            "SSIM",
            "MSE",
            "BPP",
            "Entropy",
            "Chi² (avg)",
            "RS Estimate P̂",
            "Composite Score",
            "🏆 Best at Hiding"
        ]
        
        for i, metric in enumerate(metrics):
            item = QTableWidgetItem(metric)
            item.setFlags(item.flags() & ~Qt.ItemFlag.ItemIsEditable)
            item_font = QFont()
            item_font.setBold(True)
            item.setFont(item_font)
            self.table.setItem(i, 0, item)
        
        layout.addWidget(self.table)

        self.caption = QLabel("")
        self.caption.setWordWrap(True)
        caption_font = QFont()
        caption_font.setPointSize(8)
        self.caption.setFont(caption_font)
        self.caption.setStyleSheet("color: gray;")
        layout.addWidget(self.caption)

        self.setLayout(layout)
    
    def update_results(self, result: ComparisonResult):
        """
        Update display with new comparison results.
        
        Args:
            result: ComparisonResult with AI and natural metrics
        """
        scores = result.composite_scores()
        self.comparison_result = result
        
        
        # Format chi2_mean values
        ai_chi2 = result.ai.extra.get("chi2_mean")
        nat_chi2 = result.natural.extra.get("chi2_mean")
        ai_chi2_str = f"{ai_chi2:.4f}" if ai_chi2 is not None else "N/A"
        nat_chi2_str = f"{nat_chi2:.4f}" if nat_chi2 is not None else "N/A"
        
        # Format rs_p_est values
        ai_rs = result.ai.extra.get("rs_p_est")
        nat_rs = result.natural.extra.get("rs_p_est")
        ai_rs_str = f"{ai_rs:.6f}" if ai_rs is not None else "N/A"
        nat_rs_str = f"{nat_rs:.6f}" if nat_rs is not None else "N/A"
        
        # Metric order: PSNR, SSIM, MSE, BPP, Entropy, Chi2, RS, Composite, Winner
        data = [
            (f"{result.ai.psnr:.2f}", f"{result.natural.psnr:.2f}"),
            (f"{result.ai.ssim:.6f}", f"{result.natural.ssim:.6f}"),
            (f"{result.ai.mse:.4f}", f"{result.natural.mse:.4f}"),
            (f"{result.ai.bpp:.6f}", f"{result.natural.bpp:.6f}"),
            (f"{result.ai.entropy:.4f}", f"{result.natural.entropy:.4f}"),
            (ai_chi2_str, nat_chi2_str),
            (ai_rs_str, nat_rs_str),
            (f"{scores['ai_composite']:.6f}", f"{scores['natural_composite']:.6f}"),
        ]
        
        # Fill in metric values
        for row, (ai_val, natural_val) in enumerate(data):
            ai_item = QTableWidgetItem(ai_val)
            ai_item.setFlags(ai_item.flags() & ~Qt.ItemFlag.ItemIsEditable)
            natural_item = QTableWidgetItem(natural_val)
            natural_item.setFlags(natural_item.flags() & ~Qt.ItemFlag.ItemIsEditable)
            
            self.table.setItem(row, 1, ai_item)
            self.table.setItem(row, 2, natural_item)
        
        # Winner row
        winner = result.winner()
        winner_score = result.winner_score()
        
        ai_winner = "✓" if winner == "AI" else ""
        natural_winner = "✓" if winner == "Natural" else ""
        
        ai_winner_item = QTableWidgetItem(ai_winner)
        ai_winner_item.setFlags(ai_winner_item.flags() & ~Qt.ItemFlag.ItemIsEditable)
        natural_winner_item = QTableWidgetItem(natural_winner)
        natural_winner_item.setFlags(natural_winner_item.flags() & ~Qt.ItemFlag.ItemIsEditable)
        
        # Highlight winner row
        highlight_color = QColor(200, 255, 200)  # Light green
        
        if winner == "AI":
            ai_winner_item.setBackground(highlight_color)
            natural_winner_item.setBackground(QColor(255, 240, 245))
        else:
            natural_winner_item.setBackground(highlight_color)
            ai_winner_item.setBackground(QColor(255, 240, 245))
        
        self.table.setItem(8, 1, ai_winner_item)
        self.table.setItem(8, 2, natural_winner_item)
        
        self.table.resizeRowsToContents()

        self.table.setItem(8, 1, ai_winner_item)
        self.table.setItem(8, 2, natural_winner_item)

        self.table.resizeRowsToContents()
        self.caption.setText(result.formula_caption())     # <-- add this
    
    def clear(self):
        """Clear all data."""
        for row in range(self.table.rowCount()):
            for col in range(1, 3):
                self.table.setItem(row, col, QTableWidgetItem(""))
        self.comparison_result = None

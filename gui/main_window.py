"""Main application window."""
from pathlib import Path
from typing import Optional
import cv2
from PyQt6.QtWidgets import (
    QMainWindow, QWidget, QHBoxLayout, QMessageBox, QStackedWidget,
    QMenu, QVBoxLayout, QSplitter, QScrollArea,
)
from PyQt6.QtCore import Qt

from config import IMAGE_FOLDERS
from core.facade import StegoFacade
from gui.widgets.left_panel import LeftPanel
from gui.widgets.right_panel import RightPanel
from gui.widgets.plot_dialog import PlotDialog
from gui.widgets.batch_view import BatchView
from gui.workers import BatchWorker
from gui.controllers.main_controller import MainController


class MainWindow(QMainWindow):
    """Main application window."""
    
    def __init__(self):
        """Initialize main window."""
        super().__init__()
        self.setWindowTitle("Steganography Comparison Desktop App")
        self.setGeometry(50, 50, 1400, 900)
        
        self.facade = StegoFacade()
        self.controller = MainController(self.facade)
        self.current_scene: Optional[str] = "indoor"
        self.current_natural_index: Optional[int] = None
        self.current_comparison_result = None
        
        # Batch worker
        self.batch_worker: Optional[BatchWorker] = None
        
        self.init_ui()
        self.create_menu_bar()
    
    def init_ui(self):
        """Initialize main UI layout with stacked widget for mode switching."""
        self.stacked = QStackedWidget()

        # Single Image mode: left panel (self-scrolling, pinned buttons) + right panel
        self.left_panel = LeftPanel()
        self.right_panel = RightPanel()

        splitter = QSplitter(Qt.Orientation.Horizontal)
        splitter.addWidget(self.left_panel)
        splitter.addWidget(self.right_panel)
        splitter.setStretchFactor(0, 1)
        splitter.setStretchFactor(1, 2)
        splitter.setSizes([430, 850])

        single_photo_widget = QWidget()
        single_layout = QHBoxLayout(single_photo_widget)
        single_layout.setContentsMargins(0, 0, 0, 0)
        single_layout.addWidget(splitter)

        # Batch mode
        self.batch_view = BatchView()

        self.stacked.addWidget(single_photo_widget)  # Index 0
        self.stacked.addWidget(self.batch_view)       # Index 1
        self.stacked.setCurrentIndex(0)

        self.setCentralWidget(self.stacked)
        self.connect_signals()
    
    def create_menu_bar(self):
        """Create menu bar with Mode selection."""
        menubar = self.menuBar()
        mode_menu = menubar.addMenu("Mode")
        
        # Single Image action
        self.action_single = mode_menu.addAction("Single Image")
        self.action_single.setCheckable(True)
        self.action_single.setChecked(True)
        self.action_single.triggered.connect(lambda: self.switch_mode("single"))
        
        # Batch Analysis action
        self.action_batch = mode_menu.addAction("Batch Analysis")
        self.action_batch.setCheckable(True)
        self.action_batch.triggered.connect(lambda: self.switch_mode("batch"))
    
    def switch_mode(self, mode: str):
        """Switch between single and batch modes."""
        if mode == "single":
            self.stacked.setCurrentIndex(0)
            self.action_single.setChecked(True)
            self.action_batch.setChecked(False)
        elif mode == "batch":
            self.stacked.setCurrentIndex(1)
            self.action_single.setChecked(False)
            self.action_batch.setChecked(True)
            # Refresh checklist when entering batch mode
            self.batch_view.refresh_checklist()
    
    def connect_signals(self):
        """Connect signals for single-photo mode and batch mode."""
        # Single photo mode: left panel signals
        self.left_panel.scene_changed.connect(self._on_scene_changed)
        self.left_panel.image_selected.connect(self._on_image_selected)
        self.left_panel.embed_requested.connect(self._on_embed_requested)
        self.left_panel.extract_requested.connect(self._on_extract_requested)
        self.left_panel.plots_requested.connect(self._on_plots_requested)
        
        # Controller signals
        self.controller.embed_progress.connect(self.left_panel.set_progress)
        self.controller.embed_finished.connect(self._on_embed_finished)
        self.controller.embed_error.connect(self._on_embed_error)
        self.controller.extract_progress.connect(self.left_panel.set_progress)
        self.controller.extract_finished.connect(self._on_extract_finished)
        self.controller.extract_error.connect(self._on_extract_error)
        
        # Batch mode: batch view signals
        self.batch_view.run_requested.connect(self._on_batch_run_requested)
        self.batch_view.stop_requested.connect(self._on_batch_stop_requested)
        
        # Initial load
        self._on_scene_changed("indoor")
    
    def _on_scene_changed(self, scene: str):
        """Handle scene change."""
        self.current_scene = scene
    
    def _on_image_selected(self, index: int):
        """Handle image selection."""
        self.current_natural_index = index
    
    def _on_embed_requested(self, params: dict):
        """Handle embed request."""
        kwargs = params.get('kwargs', {})
        self.controller.embed_pair(
            params['natural_path'],
            params['ai_path'],
            params['message'],
            params['strategy_name'],
            params['kb_payload'],
            **kwargs
        )
    
    def _on_embed_finished(self, result):
        """Handle embed completion."""
        self.current_comparison_result = result
        self.right_panel.metrics_table.update_results(result)
        self.left_panel.enable_extract(True)
        self.left_panel.embed_finished()
        QMessageBox.information(self, "Success", "Embedding completed!")
    
    def _on_embed_error(self, error: str):
        """Handle embed error."""
        self.left_panel.progress_bar.setVisible(False)
        self.left_panel.embed_button.setEnabled(True)
        QMessageBox.critical(self, "Embedding Error", error)
    
    def _on_extract_requested(self):
        """Handle extract request."""
        if not self.current_comparison_result:
            QMessageBox.warning(self, "Error", "No comparison result")
            return
        
        message = self.left_panel.secret_text.toPlainText().strip()
        strategy = self.left_panel.strategy_combo.currentText()
        
        # Build kwargs for strategy
        kwargs = {}
        if strategy == "LSBMR":
            kwargs['channel_idx'] = self.left_panel.channel_combo.currentIndex()
        
        self.controller.extract_pair(self.current_comparison_result, message, strategy, **kwargs)
    
    def _on_extract_finished(self, result: dict):
        """Handle extract completion."""
        self.right_panel.update_extraction(result)
        self.left_panel.progress_bar.setVisible(False)
        QMessageBox.information(self, "Success", "Extraction completed!")
    
    def _on_extract_error(self, error: str):
        """Handle extract error."""
        self.left_panel.progress_bar.setVisible(False)
        QMessageBox.critical(self, "Extraction Error", error)
    
    def _on_plots_requested(self):
        """Handle plots request."""
        if not self.current_comparison_result:
            QMessageBox.warning(self, "Error", "No comparison result")
            return
        
        strategy_name = self.left_panel.strategy_combo.currentText()
        dialog = PlotDialog(self.current_comparison_result, strategy_name, self)
        dialog.exec()
    
    def _on_batch_run_requested(self, strategies: list[str], places: list[str], 
                                 steps: list[str]):
        """Handle batch run request."""
        # Kill existing worker if any
        if self.batch_worker is not None and self.batch_worker.isRunning():
            self.batch_worker.quit()
            self.batch_worker.wait()
        
        # Create and start new worker
        self.batch_worker = BatchWorker(strategies, places, steps)
        self.batch_worker.progress.connect(self.batch_view.on_progress)
        self.batch_worker.verify_tally.connect(self.batch_view.on_verify_tally)
        self.batch_worker.finished.connect(self.batch_view.on_finished)
        self.batch_worker.error.connect(self.batch_view.on_error)
        self.batch_worker.start()
    
    def _on_batch_stop_requested(self):
        """Handle batch stop request."""
        if self.batch_worker is not None:
            self.batch_worker.request_stop()

"""Main application window left panel."""
from pathlib import Path
from PyQt6.QtWidgets import (QWidget, QVBoxLayout, QLabel, QRadioButton, QButtonGroup,
    QComboBox, QTextEdit, QPushButton, QProgressBar, QCheckBox, QFileDialog, QMessageBox)
from PyQt6.QtGui import QPixmap, QFont, QImage
from PyQt6.QtCore import pyqtSignal, Qt
import cv2
from PIL import Image as PILImage
from config import IMAGE_FOLDERS, NUM_IMAGES
from gui.widgets.image_picker import ImagePickerWidget


class LeftPanel(QWidget):
    """Left control panel."""
    
    scene_changed = pyqtSignal(str)
    image_selected = pyqtSignal(int)
    embed_requested = pyqtSignal(dict)
    extract_requested = pyqtSignal()
    plots_requested = pyqtSignal()
    
    def __init__(self, parent=None):
        """Initialize left panel."""
        super().__init__(parent)
        self.current_scene = "indoor"
        self.current_natural_index = None
        self.init_ui()
        self.connect_signals()
    
    def init_ui(self):
        """Initialize UI."""
        layout = QVBoxLayout()
        
        # Scene selection
        scene_label = QLabel("Scene Type")
        scene_label.setFont(QFont("Arial", 11, QFont.Weight.Bold))
        layout.addWidget(scene_label)
        
        scene_group = QButtonGroup()
        self.radio_indoor = QRadioButton("Indoor")
        self.radio_outdoor = QRadioButton("Outdoor")
        scene_group.addButton(self.radio_indoor)
        scene_group.addButton(self.radio_outdoor)
        layout.addWidget(self.radio_indoor)
        layout.addWidget(self.radio_outdoor)
        self.radio_indoor.setChecked(True)
        layout.addSpacing(15)
        
        # Image picker
        picker_label = QLabel("Select Image Pair")
        picker_label.setFont(QFont("Arial", 11, QFont.Weight.Bold))
        layout.addWidget(picker_label)
        
        self.image_picker = ImagePickerWidget()
        layout.addWidget(self.image_picker, stretch=1)
        
        ai_label = QLabel("AI Image Preview")
        ai_label.setFont(QFont("Arial", 10))
        ai_label.setStyleSheet("color: #666;")
        layout.addWidget(ai_label)
        
        self.ai_preview = QLabel()
        self.ai_preview.setFixedHeight(160)
        self.ai_preview.setMinimumWidth(180)
        self.ai_preview.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.ai_preview.setStyleSheet("border: 1px solid #ccc; background: #f5f5f5;")
        layout.addWidget(self.ai_preview)
        layout.addSpacing(15)
        
        # Strategy selection
        strategy_label = QLabel("Embedding Strategy")
        strategy_label.setFont(QFont("Arial", 11, QFont.Weight.Bold))
        layout.addWidget(strategy_label)
        
        from core.strategy_registry import StrategyRegistry
        self.strategy_combo = QComboBox()
        self.strategy_combo.addItems(StrategyRegistry.get_all_names())
        layout.addWidget(self.strategy_combo)
        
        # LSBMR channel selector (hidden by default)
        self.channel_label = QLabel("LSBMR Channel:")
        self.channel_label.setFont(QFont("Arial", 10))
        self.channel_label.setVisible(False)
        layout.addWidget(self.channel_label)
        
        self.channel_combo = QComboBox()
        self.channel_combo.addItems(["Blue (0)", "Green (1)", "Red (2)"])
        self.channel_combo.setCurrentIndex(0)
        self.channel_combo.setVisible(False)
        layout.addWidget(self.channel_combo)
        layout.addSpacing(15)
        
        # Payload
        payload_label = QLabel("Secret Payload")
        payload_label.setFont(QFont("Arial", 11, QFont.Weight.Bold))
        layout.addWidget(payload_label)
        
        self.payload_toggle = QCheckBox("Load from file")
        layout.addWidget(self.payload_toggle)
        
        self.file_name_label = QLabel()
        self.file_name_label.setStyleSheet("color: #0078d4; font-weight: bold;")
        self.file_name_label.setVisible(False)
        layout.addWidget(self.file_name_label)
        
        self.secret_text = QTextEdit()
        self.secret_text.setPlaceholderText("Enter message...")
        self.secret_text.setFixedHeight(80)
        layout.addWidget(self.secret_text)
        
        self.file_button = QPushButton("Browse...")
        self.file_button.setVisible(False)
        layout.addWidget(self.file_button)
        layout.addSpacing(15)
        
        # Buttons
        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        layout.addWidget(self.progress_bar)
        
        self.embed_button = QPushButton("▶ Embed")
        self.embed_button.setFont(QFont("Arial", 10, QFont.Weight.Bold))
        self.embed_button.setStyleSheet("QPushButton { background-color: #0078d4; color: white; padding: 8px; }")
        layout.addWidget(self.embed_button)
        
        self.extract_button = QPushButton("⬅ Extract")
        self.extract_button.setEnabled(False)
        layout.addWidget(self.extract_button)
        
        self.plots_button = QPushButton("📊 View Plots")
        self.plots_button.setEnabled(False)
        layout.addWidget(self.plots_button)
        
        layout.addStretch()
        self.setLayout(layout)
    
    def connect_signals(self):
        """Connect internal signals."""
        self.radio_indoor.toggled.connect(self._on_scene_changed)
        self.radio_outdoor.toggled.connect(self._on_scene_changed)
        self.image_picker.image_selected.connect(self._on_image_selected)
        self.strategy_combo.currentTextChanged.connect(self._on_strategy_changed)
        toggle = lambda: self.secret_text.setVisible(not self.payload_toggle.isChecked())
        file_btn = lambda: self.file_button.setVisible(self.payload_toggle.isChecked())
        file_label = lambda: self.file_name_label.setVisible(False) if not self.payload_toggle.isChecked() else None
        self.payload_toggle.toggled.connect(toggle)
        self.payload_toggle.toggled.connect(file_btn)
        self.payload_toggle.toggled.connect(file_label)
        self.file_button.clicked.connect(self._on_browse_payload)
        self.embed_button.clicked.connect(self._on_embed_clicked)
        self.extract_button.clicked.connect(self.extract_requested.emit)
        self.plots_button.clicked.connect(self.plots_requested.emit)
        # Trigger initial scene load for indoor
        self._on_scene_changed()
    
    def _on_scene_changed(self):
        """Handle scene type change."""
        self.current_scene = "indoor" if self.radio_indoor.isChecked() else "outdoor"
        natural_folder = IMAGE_FOLDERS[self.current_scene]["natural"]
        self.image_picker.load_images(natural_folder, NUM_IMAGES)
        self.scene_changed.emit(self.current_scene)
    
    def _on_strategy_changed(self, strategy_name: str):
        """Show/hide channel selector based on selected strategy."""
        is_lsbmr = strategy_name == "LSBMR"
        self.channel_label.setVisible(is_lsbmr)
        self.channel_combo.setVisible(is_lsbmr)
    
    def _on_image_selected(self, index: int):
        """Handle image selection."""
        self.current_natural_index = index
        ai_folder = IMAGE_FOLDERS[self.current_scene]["ai"]
        ai_images = sorted(ai_folder.glob("*.*"))
        
        if index >= len(ai_images):
            self.ai_preview.setText("AI image not found")
            return
        
        ai_path = ai_images[index]
        try:
            bgr = cv2.imread(str(ai_path))
            if bgr is not None:
                rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
                h, w = rgb.shape[:2]
                bytes_per_line = 3 * w
                # Create QImage from full resolution
                qimg = QImage(rgb.data, w, h, bytes_per_line,
                             QImage.Format.Format_RGB888).copy()
                qpixmap = QPixmap.fromImage(qimg)
                # Scale with high-quality SmoothTransformation
                scaled = qpixmap.scaledToHeight(160, 
                                               Qt.TransformationMode.SmoothTransformation)
                self.ai_preview.setPixmap(scaled)
            else:
                self.ai_preview.setText("Failed to load image")
        except Exception as e:
            self.ai_preview.setText("Preview failed")
        self.image_selected.emit(index)
    
    def _on_browse_payload(self):
        """Browse for payload file."""
        path, _ = QFileDialog.getOpenFileName(self, "Select Secret File", "", "Text Files (*.txt)")
        if path:
            try:
                with open(path, 'r') as f:
                    content = f.read()
                    self.secret_text.setText(content)
                # Show filename in blue to indicate file is loaded
                file_name = Path(path).name
                self.file_name_label.setText(f"✓ Loaded: {file_name}")
                self.file_name_label.setVisible(True)
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Failed: {e}")
                self.file_name_label.setText("Failed to load file")
                self.file_name_label.setVisible(True)
    
    def _on_embed_clicked(self):
        """Prepare and emit embed request."""
        if self.current_natural_index is None:
            QMessageBox.warning(self, "Error", "Select image")
            return
        
        message = self.secret_text.toPlainText().strip()
        if not message:
            QMessageBox.warning(self, "Error", "Enter message")
            return
        
        natural_path = self.image_picker.get_selected_image_path()
        ai_folder = IMAGE_FOLDERS[self.current_scene]["ai"]
        ai_images = sorted(ai_folder.glob("*.*"))
        
        if self.current_natural_index >= len(ai_images):
            QMessageBox.critical(self, "Error", "AI image not found")
            return
        
        ai_path = ai_images[self.current_natural_index]
        kb_payload = len(message.encode('utf-8')) / 1024
        
        self.progress_bar.setVisible(True)
        self.progress_bar.setValue(0)
        self.embed_button.setEnabled(False)
        
        # Build kwargs for strategy
        kwargs = {}
        strategy_name = self.strategy_combo.currentText()
        if strategy_name == "LSBMR":
            kwargs['channel_idx'] = self.channel_combo.currentIndex()
        
        self.embed_requested.emit({
            'natural_path': natural_path, 'ai_path': ai_path, 'message': message,
            'strategy_name': strategy_name, 'kb_payload': kb_payload,
            'kwargs': kwargs,
        })
    
    def enable_extract(self, enabled: bool):
        """Enable/disable extract button."""
        self.extract_button.setEnabled(enabled)
        self.plots_button.setEnabled(enabled)
    
    def set_progress(self, value: int):
        """Update progress bar."""
        self.progress_bar.setValue(value)
    
    def embed_finished(self):
        """Reset after embed."""
        self.progress_bar.setVisible(False)
        self.embed_button.setEnabled(True)

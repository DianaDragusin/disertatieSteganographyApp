"""Image picker widget with thumbnail grid."""
from pathlib import Path
from typing import Optional, List
from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QScrollArea, QLabel,
    QPushButton, QFrame
)
from PyQt6.QtCore import pyqtSignal, Qt, QSize
from PyQt6.QtGui import QPixmap, QIcon, QImage
from PIL import Image
import cv2
import numpy as np

from config import IMAGE_EXTENSIONS


class ThumbnailButton(QPushButton):
    """Single thumbnail button for image selection."""
    
    def __init__(self, image_path: Path, index: int, size: int = 120):
        """
        Initialize thumbnail button.
        
        Args:
            image_path: Path to image file
            index: Image index (0-19)
            size: Thumbnail size in pixels
        """
        super().__init__()
        self.image_path = image_path
        self.index = index
        self.size = size
        self.is_selected = False
        
        self.setFixedSize(size + 10, size + 10)
        self.load_thumbnail()
        self.clicked.connect(self._on_click)
    
    def load_thumbnail(self):
        """Load and display thumbnail."""
        try:
            bgr = cv2.imread(str(self.image_path))
            if bgr is not None:
                rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
                # Resize for thumbnail
                h, w = rgb.shape[:2]
                aspect = w / h
                if aspect > 1:
                    new_w = self.size
                    new_h = int(self.size / aspect)
                else:
                    new_h = self.size
                    new_w = int(self.size * aspect)
                rgb = cv2.resize(rgb, (new_w, new_h))
                
                # Create QImage directly from numpy with .copy() for garbage collection
                h_thumb, w_thumb = rgb.shape[:2]
                bytes_per_line = 3 * w_thumb
                qimg = QImage(rgb.data, w_thumb, h_thumb, bytes_per_line, 
                             QImage.Format.Format_RGB888).copy()
                pixmap = QPixmap.fromImage(qimg).scaledToHeight(
                    self.size, Qt.TransformationMode.SmoothTransformation)
                self.setIcon(QIcon(pixmap))
                self.setIconSize(QSize(self.size, self.size))
        except Exception as e:
            print(f"[ThumbnailButton] Error loading {self.image_path}: {e}")
            pass
    
    def _on_click(self):
        """Handle thumbnail click."""
        self.select()
    
    def select(self):
        """Mark as selected."""
        self.is_selected = True
        self.setStyleSheet("border: 3px solid #0078d4; border-radius: 8px;")
    
    def deselect(self):
        """Mark as deselected."""
        self.is_selected = False
        self.setStyleSheet("")


class ImagePickerWidget(QWidget):
    """Grid widget for picking images from a folder."""
    
    image_selected = pyqtSignal(int)  # Emits index of selected image
    
    def __init__(self, parent=None):
        """Initialize image picker."""
        super().__init__(parent)
        self.thumbnails: List[ThumbnailButton] = []
        self.current_index: Optional[int] = None
        self.image_folder: Optional[Path] = None
        
        self.init_ui()
    
    def init_ui(self):
        """Initialize UI layout."""
        layout = QVBoxLayout()
        
        # Title
        title = QLabel("Select Image:")
        title.setStyleSheet("font-weight: bold; font-size: 12px;")
        layout.addWidget(title)
        
        # Scroll area with grid of thumbnails
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setStyleSheet("QScrollArea { border: 1px solid #ccc; }")
        
        grid_widget = QWidget()
        grid_layout = QHBoxLayout(grid_widget)
        grid_layout.setSpacing(8)
        grid_layout.setContentsMargins(5, 5, 5, 5)
        grid_layout.addStretch()
        
        self.grid_layout = grid_layout
        scroll.setWidget(grid_widget)
        
        layout.addWidget(scroll, stretch=1)
        self.setLayout(layout)
    
    def load_images(self, folder: Path, num_images: int = 20) -> bool:
        """
        Load images from folder.
        
        Args:
            folder: Path to folder containing images
            num_images: Expected number of images
            
        Returns:
            True if successful, False otherwise
        """
        if not folder.exists():
            return False
        
        self.image_folder = folder
        
        # Find image files with case-insensitive extension matching
        image_files = []
        for item in sorted(folder.iterdir()):
            if item.is_file() and item.suffix.lower() in {ext.lower() for ext in IMAGE_EXTENSIONS}:
                image_files.append(item)
        
        image_files = image_files[:num_images]
        
        if not image_files:
            print(f"[ImagePicker] No images found in {folder}")
            return False
        
        print(f"[ImagePicker] Loaded {len(image_files)} thumbs from {folder}")
        
        # Clear previous thumbnails
        for thumb in self.thumbnails:
            thumb.deleteLater()
        self.thumbnails.clear()
        
        # Add new thumbnails
        for idx, img_path in enumerate(image_files):
            thumb = ThumbnailButton(img_path, idx, size=150)
            thumb.clicked.connect(lambda checked=False, i=idx: self._on_thumbnail_selected(i))
            self.thumbnails.append(thumb)
            self.grid_layout.insertWidget(len(self.thumbnails) - 1, thumb)
        
        return True
    
    def _on_thumbnail_selected(self, index: int):
        """Handle thumbnail selection."""
        if self.current_index is not None and self.current_index < len(self.thumbnails):
            self.thumbnails[self.current_index].deselect()
        
        self.current_index = index
        self.thumbnails[index].select()
        self.image_selected.emit(index)
    
    def select_image(self, index: int):
        """Programmatically select an image."""
        if 0 <= index < len(self.thumbnails):
            self._on_thumbnail_selected(index)
    
    def get_selected_image_path(self) -> Optional[Path]:
        """Get path of currently selected image."""
        if self.current_index is not None and self.image_folder:
            return self.thumbnails[self.current_index].image_path
        return None

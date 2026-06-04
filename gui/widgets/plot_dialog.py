"""Dialog for viewing steganography comparison plots."""
import numpy as np
import cv2
from PyQt6.QtWidgets import QDialog, QVBoxLayout, QTabWidget, QWidget
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from scipy.stats import pearsonr
from core.models import ComparisonResult
from extraction_tracking import build_extraction_report, render_extraction_map


class PlotDialog(QDialog):
    """Dialog with four tabbed matplotlib plots comparing AI vs Natural embedding."""
    
    # Colors for consistent styling
    AI_COLOR = "#3498db"          # Blue
    NATURAL_COLOR = "#e67e22"     # Orange
    COVER_COLOR = "#3498db"       # Blue for histograms
    STEGO_COLOR = "#e67e22"       # Orange for histograms
    
    def __init__(self, comparison_result: ComparisonResult, strategy_name: str, parent=None):
        """
        Initialize plot dialog.
        
        Args:
            comparison_result: ComparisonResult with ai and natural MetricsResult objects
            strategy_name: Name of the embedding strategy (e.g., "LSB Random Spatial")
            parent: Parent widget
        """
        super().__init__(parent)
        self.result = comparison_result
        self.strategy_name = strategy_name
        
        # Extract arrays from the comparison result
        self.ai_cover = self.result.ai.extra.get("cover_array").astype(np.uint8)
        self.ai_stego = self.result.ai.extra.get("stego_array").astype(np.uint8)
        self.natural_cover = self.result.natural.extra.get("cover_array").astype(np.uint8)
        self.natural_stego = self.result.natural.extra.get("stego_array").astype(np.uint8)
        
        # Determine if grayscale or color
        self.is_grayscale = self.ai_cover.ndim == 2
        
        self.setWindowTitle(f"Analysis Plots — {strategy_name}")
        self.setGeometry(100, 100, 1100, 700)
        
        self.init_ui()
    
    def init_ui(self):
        """Initialize UI with tabbed plots."""
        layout = QVBoxLayout()
        
        # Tab widget
        tabs = QTabWidget()
        
        # Tab 1: Difference Heatmap
        tab1 = QWidget()
        tab1_layout = QVBoxLayout(tab1)
        canvas1 = self._create_difference_heatmap_canvas()
        tab1_layout.addWidget(canvas1)
        tabs.addTab(tab1, "Difference Heatmap")
        
        # Tab 2: Pair-Difference Histogram
        tab2 = QWidget()
        tab2_layout = QVBoxLayout(tab2)
        canvas2 = self._create_histogram_canvas()
        tab2_layout.addWidget(canvas2)
        tabs.addTab(tab2, "Pair-Difference Histogram")
        
        # Tab 3: Channel Modification
        tab3 = QWidget()
        tab3_layout = QVBoxLayout(tab3)
        canvas3 = self._create_channel_modification_canvas()
        tab3_layout.addWidget(canvas3)
        tabs.addTab(tab3, "Channel Modification")
        
        # Tab 4: Metric Comparison
        tab4 = QWidget()
        tab4_layout = QVBoxLayout(tab4)
        canvas4 = self._create_metric_comparison_canvas()
        tab4_layout.addWidget(canvas4)
        tabs.addTab(tab4, "Metric Comparison")

        # Tab: Extraction Map
        tab_extract = QWidget()
        tab_extract_layout = QVBoxLayout(tab_extract)
        tab_extract_layout.addWidget(self._create_extraction_map_canvas())
        tabs.addTab(tab_extract, "Extraction Map")
        
        # Tab 5: Channel Correlation (only for color images)
        if not self.is_grayscale:
            tab5 = QWidget()
            tab5_layout = QVBoxLayout(tab5)
            canvas5 = self._create_channel_correlation_canvas()
            tab5_layout.addWidget(canvas5)
            tabs.addTab(tab5, "Channel Correlation")
        
        layout.addWidget(tabs)  
        self.setLayout(layout)
    
    def _create_difference_heatmap_canvas(self) -> FigureCanvas:
        """Create heatmap showing |cover - stego| differences."""
        figure = Figure(figsize=(12, 5), dpi=100)
        
        # AI subplot
        ax_ai = figure.add_subplot(121)
        diff_ai = np.abs(self.ai_cover.astype(int) - self.ai_stego.astype(int))
        if not self.is_grayscale:
            diff_ai = np.max(diff_ai, axis=2)
        im_ai = ax_ai.imshow(diff_ai, cmap="hot")
        ax_ai.set_title("AI Image", fontsize=11, fontweight='bold')
        ax_ai.axis('off')
        figure.colorbar(im_ai, ax=ax_ai, label="Pixel difference")
        
        # Natural subplot
        ax_nat = figure.add_subplot(122)
        diff_nat = np.abs(self.natural_cover.astype(int) - self.natural_stego.astype(int))
        if not self.is_grayscale:
            diff_nat = np.max(diff_nat, axis=2)
        im_nat = ax_nat.imshow(diff_nat, cmap="hot")
        ax_nat.set_title("Natural Image", fontsize=11, fontweight='bold')
        ax_nat.axis('off')
        figure.colorbar(im_nat, ax=ax_nat, label="Pixel difference")
        
        figure.suptitle("Where pixels were modified", fontsize=12, fontweight='bold')
        figure.tight_layout()
        
        canvas = FigureCanvas(figure)
        return canvas

    def _create_extraction_map_canvas(self) -> FigureCanvas:
        figure = Figure(figsize=(12, 6), dpi=100)
        panels = [
            ("AI", self.ai_stego, self.result.ai.extra, 121),
            ("Natural", self.natural_stego, self.result.natural.extra, 122),
        ]
        for label, stego, extra, pos in panels:
            ax = figure.add_subplot(pos)
            try:
                message = extra.get("message")
                if not message:
                    raise ValueError("No message stored from embedding.")
                report = build_extraction_report(
                    self.strategy_name, stego, message,
                    coordinate_key=extra.get("lsbmr_key"),
                    channel_idx=extra.get("channel_idx", 0),
                )
                render_extraction_map(ax, stego, report, title=label)
            except Exception as e:
                ax.imshow(
                    cv2.cvtColor(stego, cv2.COLOR_BGR2GRAY) if stego.ndim == 3 else stego,
                    cmap="gray", vmin=0, vmax=255,
                )
                ax.set_title(f"{label}: extraction map unavailable\n{e}", fontsize=9)
                ax.axis("off")
        figure.tight_layout()
        return FigureCanvas(figure)
    
    def _create_histogram_canvas(self) -> FigureCanvas:
        """Create histogram of neighbor pixel differences (cover vs stego)."""
        figure = Figure(figsize=(12, 5), dpi=100)
        
        # Helper function to compute neighbor differences
        def compute_neighbor_diffs(img):
            """Compute horizontal neighbor differences for an image."""
            if img.ndim == 3:
                img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            diffs = np.abs(img[:, 1:].astype(np.int16) - img[:, :-1].astype(np.int16)).flatten()
            return diffs
        
        # AI subplot
        ax_ai = figure.add_subplot(121)
        ai_cover_diffs = compute_neighbor_diffs(self.ai_cover)
        ai_stego_diffs = compute_neighbor_diffs(self.ai_stego)
        
        counts_cover, bins = np.histogram(ai_cover_diffs, bins=np.arange(0, 65), density=True)
        counts_stego, _ = np.histogram(ai_stego_diffs, bins=np.arange(0, 65), density=True)
        bin_centers = (bins[:-1] + bins[1:]) / 2
        
        ax_ai.plot(bin_centers, counts_cover, color=self.COVER_COLOR, label="Cover", linewidth=2)
        ax_ai.plot(bin_centers, counts_stego, color=self.STEGO_COLOR, label="Stego", linewidth=2)
        ax_ai.set_title("AI Image", fontsize=11, fontweight='bold')
        ax_ai.set_xlabel("Pixel pair difference", fontsize=10)
        ax_ai.set_ylabel("Density", fontsize=10)
        ax_ai.legend(fontsize=9)
        ax_ai.grid(True, alpha=0.3)
        
        # Natural subplot
        ax_nat = figure.add_subplot(122)
        nat_cover_diffs = compute_neighbor_diffs(self.natural_cover)
        nat_stego_diffs = compute_neighbor_diffs(self.natural_stego)
        
        counts_cover_nat, bins_nat = np.histogram(nat_cover_diffs, bins=np.arange(0, 65), density=True)
        counts_stego_nat, _ = np.histogram(nat_stego_diffs, bins=np.arange(0, 65), density=True)
        bin_centers_nat = (bins_nat[:-1] + bins_nat[1:]) / 2
        
        ax_nat.plot(bin_centers_nat, counts_cover_nat, color=self.COVER_COLOR, label="Cover", linewidth=2)
        ax_nat.plot(bin_centers_nat, counts_stego_nat, color=self.STEGO_COLOR, label="Stego", linewidth=2)
        ax_nat.set_title("Natural Image", fontsize=11, fontweight='bold')
        ax_nat.set_xlabel("Pixel pair difference", fontsize=10)
        ax_nat.set_ylabel("Density", fontsize=10)
        ax_nat.legend(fontsize=9)
        ax_nat.grid(True, alpha=0.3)
        
        figure.suptitle("Distribution of neighbor pixel differences", fontsize=12, fontweight='bold')
        figure.tight_layout()
        
        canvas = FigureCanvas(figure)
        return canvas
    
    def _create_channel_modification_canvas(self) -> FigureCanvas:
        """Create bar chart showing modified pixels per channel."""
        figure = Figure(figsize=(10, 5), dpi=100)
        ax = figure.add_subplot(111)
        
        if self.is_grayscale:
            # Grayscale: single channel
            ai_modified = np.sum(self.ai_cover != self.ai_stego)
            nat_modified = np.sum(self.natural_cover != self.natural_stego)
            
            channels = ["Grayscale"]
            ai_counts = [ai_modified]
            nat_counts = [nat_modified]
        else:
            # Color: three channels
            channels = ["Blue", "Green", "Red"]
            ai_counts = []
            nat_counts = []
            
            for ch in range(3):
                ai_modified = np.sum(self.ai_cover[:, :, ch] != self.ai_stego[:, :, ch])
                nat_modified = np.sum(self.natural_cover[:, :, ch] != self.natural_stego[:, :, ch])
                ai_counts.append(ai_modified)
                nat_counts.append(nat_modified)
        
        # Create grouped bar chart
        x = np.arange(len(channels))
        width = 0.35
        
        bars1 = ax.bar(x - width/2, ai_counts, width, label="AI", color=self.AI_COLOR, alpha=0.8)
        bars2 = ax.bar(x + width/2, nat_counts, width, label="Natural", color=self.NATURAL_COLOR, alpha=0.8)
        
        ax.set_xlabel("Channel", fontsize=10, fontweight='bold')
        ax.set_ylabel("Pixels modified (count)", fontsize=10, fontweight='bold')
        ax.set_title("Modifications per channel", fontsize=12, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(channels)
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3, axis='y')
        
        figure.tight_layout()
        
        canvas = FigureCanvas(figure)
        return canvas
    
    def _create_metric_comparison_canvas(self) -> FigureCanvas:
        """Create bar chart comparing quality metrics."""
        figure = Figure(figsize=(10, 5), dpi=100)
        ax = figure.add_subplot(111)
        
        # Get metrics
        ai_psnr = self.result.ai.psnr
        ai_ssim = self.result.ai.ssim
        ai_mse = self.result.ai.mse
        
        nat_psnr = self.result.natural.psnr
        nat_ssim = self.result.natural.ssim
        nat_mse = self.result.natural.mse
        
        # Scale for visualization
        metrics_labels = ["PSNR (dB)", "SSIM × 100", "MSE × 100"]
        ai_values = [ai_psnr, ai_ssim * 100, ai_mse * 100]
        nat_values = [nat_psnr, nat_ssim * 100, nat_mse * 100]
        
        # Create grouped bar chart
        x = np.arange(len(metrics_labels))
        width = 0.35
        
        bars1 = ax.bar(x - width/2, ai_values, width, label="AI", color=self.AI_COLOR, alpha=0.8)
        bars2 = ax.bar(x + width/2, nat_values, width, label="Natural", color=self.NATURAL_COLOR, alpha=0.8)
        
        # Add value labels on bars (unscaled)
        def add_labels(bars, unscaled_values):
            for bar, val in zip(bars, unscaled_values):
                height = bar.get_height()
                # Determine format based on metric
                if val < 10:  # SSIM and MSE
                    fmt = f'{val:.4f}'
                else:  # PSNR
                    fmt = f'{val:.2f}'
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       fmt, ha='center', va='bottom', fontsize=9)
        
        add_labels(bars1, [ai_psnr, ai_ssim, ai_mse])
        add_labels(bars2, [nat_psnr, nat_ssim, nat_mse])
        
        ax.set_ylabel("Metric value (scaled)", fontsize=10, fontweight='bold')
        ax.set_title("Quality metrics — AI vs Natural", fontsize=12, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(metrics_labels)
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3, axis='y')
        
        figure.tight_layout()
        
        canvas = FigureCanvas(figure)
        return canvas
    
    def _create_channel_correlation_canvas(self) -> FigureCanvas:
        """Create grouped bar chart showing inter-channel Pearson correlations."""
        figure = Figure(figsize=(12, 5), dpi=100)
        
        # Channel pair labels and indices (BGR format: B=0, G=1, R=2)
        pair_labels = ["R-G", "R-B", "G-B"]
        pair_indices = [(2, 1), (2, 0), (1, 0)]  # (R,G), (R,B), (G,B) in BGR
        
        def compute_correlations(cover, stego):
            """Compute Pearson correlations for channel pairs."""
            try:
                correlations_cover = []
                correlations_stego = []
                
                for ch1_idx, ch2_idx in pair_indices:
                    ch1_cover = cover[:, :, ch1_idx].astype(np.float32).flatten()
                    ch2_cover = cover[:, :, ch2_idx].astype(np.float32).flatten()
                    r_cover, _ = pearsonr(ch1_cover, ch2_cover)
                    correlations_cover.append(r_cover)
                    
                    ch1_stego = stego[:, :, ch1_idx].astype(np.float32).flatten()
                    ch2_stego = stego[:, :, ch2_idx].astype(np.float32).flatten()
                    r_stego, _ = pearsonr(ch1_stego, ch2_stego)
                    correlations_stego.append(r_stego)
                
                return correlations_cover, correlations_stego
            except Exception as e:
                print(f"Error computing correlations: {e}")
                return [0, 0, 0], [0, 0, 0]
        
        # AI subplot
        ax_ai = figure.add_subplot(121)
        ai_cover_corr, ai_stego_corr = compute_correlations(self.ai_cover, self.ai_stego)
        
        x = np.arange(len(pair_labels))
        width = 0.35
        
        bars_ai_cover = ax_ai.bar(x - width/2, ai_cover_corr, width, label="Cover", 
                                   color=self.COVER_COLOR, alpha=0.8)
        bars_ai_stego = ax_ai.bar(x + width/2, ai_stego_corr, width, label="Stego", 
                                   color=self.STEGO_COLOR, alpha=0.8)
        
        # Annotate bars with correlation values
        for bar, val in zip(bars_ai_cover, ai_cover_corr):
            height = bar.get_height()
            ax_ai.text(bar.get_x() + bar.get_width()/2., height,
                      f'{val:.4f}', ha='center', va='bottom', fontsize=8)
        for bar, val in zip(bars_ai_stego, ai_stego_corr):
            height = bar.get_height()
            ax_ai.text(bar.get_x() + bar.get_width()/2., height,
                      f'{val:.4f}', ha='center', va='bottom', fontsize=8)
        
        ax_ai.set_title("AI Image", fontsize=11, fontweight='bold')
        ax_ai.set_ylabel("Pearson r", fontsize=10, fontweight='bold')
        ax_ai.set_xticks(x)
        ax_ai.set_xticklabels(pair_labels)
        ax_ai.set_ylim([-0.1, 1.1])
        ax_ai.legend(fontsize=9)
        ax_ai.grid(True, alpha=0.3, axis='y')
        
        # Compute average |Δr| for AI
        avg_delta_r_ai = np.mean(np.abs(np.array(ai_cover_corr) - np.array(ai_stego_corr)))
        ax_ai.text(0.5, -0.25, f'Avg |Δr| = {avg_delta_r_ai:.4f}', 
                  transform=ax_ai.transAxes, ha='center', fontsize=9, style='italic')
        
        # Natural subplot
        ax_nat = figure.add_subplot(122)
        nat_cover_corr, nat_stego_corr = compute_correlations(self.natural_cover, self.natural_stego)
        
        bars_nat_cover = ax_nat.bar(x - width/2, nat_cover_corr, width, label="Cover", 
                                     color=self.COVER_COLOR, alpha=0.8)
        bars_nat_stego = ax_nat.bar(x + width/2, nat_stego_corr, width, label="Stego", 
                                     color=self.STEGO_COLOR, alpha=0.8)
        
        # Annotate bars with correlation values
        for bar, val in zip(bars_nat_cover, nat_cover_corr):
            height = bar.get_height()
            ax_nat.text(bar.get_x() + bar.get_width()/2., height,
                       f'{val:.4f}', ha='center', va='bottom', fontsize=8)
        for bar, val in zip(bars_nat_stego, nat_stego_corr):
            height = bar.get_height()
            ax_nat.text(bar.get_x() + bar.get_width()/2., height,
                       f'{val:.4f}', ha='center', va='bottom', fontsize=8)
        
        ax_nat.set_title("Natural Image", fontsize=11, fontweight='bold')
        ax_nat.set_ylabel("Pearson r", fontsize=10, fontweight='bold')
        ax_nat.set_xticks(x)
        ax_nat.set_xticklabels(pair_labels)
        ax_nat.set_ylim([-0.1, 1.1])
        ax_nat.legend(fontsize=9)
        ax_nat.grid(True, alpha=0.3, axis='y')
        
        # Compute average |Δr| for Natural
        avg_delta_r_nat = np.mean(np.abs(np.array(nat_cover_corr) - np.array(nat_stego_corr)))
        ax_nat.text(0.5, -0.25, f'Avg |Δr| = {avg_delta_r_nat:.4f}', 
                   transform=ax_nat.transAxes, ha='center', fontsize=9, style='italic')
        
        figure.suptitle("Inter-channel correlation — Cover vs Stego", fontsize=12, fontweight='bold')
        figure.tight_layout()
        
        canvas = FigureCanvas(figure)
        return canvas

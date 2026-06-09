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
        
        # Tab 2: Pixel Frequency Histogram
        tab2 = QWidget()
        tab2_layout = QVBoxLayout(tab2)
        canvas2 = self._create_histogram_canvas()
        tab2_layout.addWidget(canvas2)
        tabs.addTab(tab2, "Pixel Frequency Histogram")


        # --- NEW TAB: RS Group Differences ---
        tab_rs_diff = QWidget()
        tab_rs_diff_layout = QVBoxLayout(tab_rs_diff)
        tab_rs_diff_layout.addWidget(self._create_rs_group_differences_canvas())
        tabs.addTab(tab_rs_diff, "RS Group Differences")
        
        # Tab 3: Channel Modification
        tab3 = QWidget()
        tab3_layout = QVBoxLayout(tab3)
        canvas3 = self._create_channel_modification_canvas()
        tab3_layout.addWidget(canvas3)
        tabs.addTab(tab3, "Channel Modification")

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
        """
        Pixel value histogram, before vs after embedding.

        For each panel (AI vs Natural) it overlays two intensity histograms:
          - gray (filled)  : the cover values, BEFORE embedding
          - colour (line)  : the stego values, AFTER embedding

        So you can read how the value counts shift once the payload is written.

        The analysis plane depends on the strategy:
          - grayscale strategies (PVD)            -> the grayscale plane
          - LSBMR                                 -> the single channel the
                                                     payload was embedded into
          - other colour strategies (LSB Random,
            Canny-Sobel)                          -> the luminance plane
        """
        figure = Figure(figsize=(12, 5), dpi=100)

        BEFORE_COLOR = "#9e9e9e"  # gray: cover (before)
        AFTER_COLOR = "#e67e22"   # orange: stego (after)

        name = (self.strategy_name or "").lower()
        is_lsbmr = "lsbmr" in name
        ch_names = {0: "blue", 1: "green", 2: "red"}  # OpenCV BGR order

        def plane_of(arr, extra):
            """Return (intensity_plane_2d, label) for this strategy."""
            if arr.ndim == 2:
                return arr, "grayscale"
            if is_lsbmr:
                ch = int(extra.get("channel_idx", 0))
                return arr[:, :, ch], ch_names.get(ch, f"ch{ch}")
            return cv2.cvtColor(arr, cv2.COLOR_BGR2GRAY), "luminance"

        def panel(ax, cover, stego, extra, title):
            cov_plane, sub = plane_of(cover, extra)
            ste_plane, _ = plane_of(stego, extra)
            bins = np.arange(0, 257)  # one bin per intensity value 0..255
            centers = (bins[:-1] + bins[1:]) / 2

            # Cover: filled gray bars (the "how many pixels have each value")
            ax.hist(cov_plane.ravel(), bins=bins, color=BEFORE_COLOR,
                    label="Before (cover)")

            # Stego: orange outline on top (the values after embedding)
            counts_after, _ = np.histogram(ste_plane.ravel(), bins=bins)
            ax.step(centers, counts_after, where='mid', color=AFTER_COLOR,
                    linewidth=1.2, label="After (stego)")

            ax.set_title(f"{title}  ({sub})", fontsize=11, fontweight='bold')
            ax.set_xlabel("Pixel intensity", fontsize=10)
            ax.set_ylabel("Frequency (pixel count)", fontsize=10)
            ax.set_xlim(0, 255)
            ax.legend(fontsize=9)
            ax.grid(True, alpha=0.3, axis='y')

        ax_ai = figure.add_subplot(121)
        panel(ax_ai, self.ai_cover, self.ai_stego, self.result.ai.extra, "AI Image")

        ax_nat = figure.add_subplot(122)
        panel(ax_nat, self.natural_cover, self.natural_stego, self.result.natural.extra, "Natural Image")

        figure.suptitle("Pixel value histogram — before vs after embedding",
                        fontsize=12, fontweight='bold')
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

    def _create_rs_group_differences_canvas(self) -> FigureCanvas:
        """
        Plots the frequency distribution of the discrimination function values 
        (local pixel group differences) to compare AI vs Natural image texture profiles.
        """
        figure = Figure(figsize=(10, 5), dpi=100)
        ax = figure.add_subplot(111)

        group_size = 4
        name = (self.strategy_name or "").lower()
        is_lsbmr = "lsbmr" in name

        def get_discrimination_values(arr, extra):
            """Extracts the appropriate channel and returns its group discrimination vector."""
            if arr.ndim == 3:
                if is_lsbmr:
                    ch = int(extra.get("channel_idx", 0))
                    channel = arr[:, :, ch]
                else:
                    channel = cv2.cvtColor(arr, cv2.COLOR_BGR2GRAY)
            else:
                channel = arr
                
            flat = channel.flatten()
            num_groups = len(flat) // group_size
            groups = flat[:num_groups * group_size].reshape(num_groups, group_size)
            
            # Vectorized discrimination calculation: sum of absolute differences between adjacent pixels in groups
            return np.sum(np.abs(np.diff(groups.astype(np.int32), axis=1)), axis=1)

        # Compute local variations for both cover images
        ai_diffs = get_discrimination_values(self.ai_cover, self.result.ai.extra)
        nat_diffs = get_discrimination_values(self.natural_cover, self.result.natural.extra)

        # Establish identical binning for both to guarantee matching coordinates
        if len(ai_diffs) > 0 and len(nat_diffs) > 0:
            max_val = max(np.max(ai_diffs), np.max(nat_diffs))
            # Caps the range reasonably for readability (max difference of 4-pixel block could theoretically be 3*255=765)
            bins = np.arange(0, min(max_val + 2, 250)) 
        else:
            bins = np.arange(0, 100)

        # Plot using 'step' histograms with density=True to normalize for differing image resolutions
        ax.hist(ai_diffs, bins=bins, histtype='step', linewidth=2.0, 
                color=self.AI_COLOR, label="AI Image (Cover)", density=True)
        ax.hist(nat_diffs, bins=bins, histtype='step', linewidth=2.0, 
                color=self.NATURAL_COLOR, label="Natural Image (Cover)", density=True)

        # Labeling and styling
        ax.set_title("Local Pixel Group Variance: AI vs Natural Cover Images\n"
                     "(Calculated via RS Discrimination Function)", fontsize=11, fontweight='bold')
        ax.set_xlabel("Discrimination value $f(G)$ (Sum of Absolute Differences)", fontsize=10)
        ax.set_ylabel("Probability Density (Normalized Frequency)", fontsize=10)
        ax.legend(fontsize=10, loc="upper right")
        ax.grid(True, alpha=0.3)
        
        # Crop x-axis dynamically based on data spread to eliminate trailing empty space
        if len(ai_diffs) > 0 and len(nat_diffs) > 0:
            high_percentile = max(np.percentile(ai_diffs, 99), np.percentile(nat_diffs, 99))
            ax.set_xlim(0, high_percentile + 10)

        figure.tight_layout()
        return FigureCanvas(figure)
    
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
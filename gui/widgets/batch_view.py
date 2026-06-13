"""
gui/widgets/batch_view.py

The Batch Analysis mode page: viewer + optional runner.
"""

from __future__ import annotations

import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QGroupBox, QLabel, QPushButton,
    QCheckBox, QProgressBar, QScrollArea, QTabWidget, QMessageBox,
    QComboBox, QTextEdit, QGridLayout, QTableWidget, QTableWidgetItem,
    QHeaderView, QSplitter,
)
from PyQt6.QtCore import pyqtSignal, Qt
from PyQt6.QtGui import QColor
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas

from gui.batch_results_index import scan_results, load_audit_detail, format_audit_pass_rate, get_status_icon
from gui.widgets.batch_plots import (
    fig_quality_2x2, fig_chi2_2x2, fig_rs_2x2, fig_pairdiff_2x2,
)
from gui.widgets.batch_report import (
    aggregate, fig_rank_heatmap, build_verdict_text,
)
from batch_runner import STRATEGY_TABLE


def _pairdiff_placeholder_figure(message: str = "") -> Figure:
    """Lightweight placeholder shown until the pair-diff tab is opened."""
    fig = Figure(figsize=(12, 8))
    fig.suptitle("Pair-Difference Histograms", fontsize=14, fontweight="bold")
    ax = fig.add_subplot(111)
    text = message or ("Click this tab to compute pair-difference histograms.\n"
                       "Reading stego images from disk takes a few seconds.")
    ax.text(0.5, 0.5, text, ha="center", va="center",
            transform=ax.transAxes, fontsize=11, color="gray")
    ax.set_xticks([])
    ax.set_yticks([])
    return fig


class BatchView(QWidget):
    """Main widget for batch analysis mode."""

    run_requested = pyqtSignal(list, list, list)  # strategies, places, steps
    stop_requested = pyqtSignal()

    def __init__(self, parent=None):
        """Initialize batch view."""
        super().__init__(parent)
        self.current_scan = {}
        self.current_agg = None
        self.init_ui()
        self.refresh_checklist()

    def init_ui(self):
        """Initialize UI layout as two top-level tabs:
        - 'Run Pipeline': the run-from-zero controls + execution progress
        - 'Results': the CSV-derived plots and report, with the full height
        """
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(0, 0, 0, 0)

        self.mode_tabs = QTabWidget()

        # ── Tab 1: Run Pipeline (controls + progress only) ─────────────────
        run_tab = QWidget()
        run_layout = QVBoxLayout(run_tab)
        run_layout.addWidget(self._build_top_section(), stretch=1)
        run_layout.addWidget(self._build_progress_section())
        run_tab.setLayout(run_layout)
        self.mode_tabs.addTab(run_tab, "Run Pipeline")

        # ── Tab 2: Results (plots + report get the whole window) ───────────
        self.tabs = QTabWidget()
        self._build_result_tabs()
        self.mode_tabs.addTab(self.tabs, "Results")

        main_layout.addWidget(self.mode_tabs)
        self.setLayout(main_layout)

    def _build_top_section(self) -> QWidget:
        """Build the master/detail table layout and run/stop controls."""
        widget = QGroupBox("Audit Coverage & Pipeline Control")
        layout = QVBoxLayout()

        # ── Master/Detail tables (left/right split) ──────────────────────────
        tables_splitter = QSplitter(Qt.Orientation.Horizontal)

        # Master table (left)
        master_group = QGroupBox("Results Summary")
        master_layout = QVBoxLayout()
        self.master_table = QTableWidget()
        self.master_table.setColumnCount(8)
        self.master_table.setHorizontalHeaderLabels(
            ["Method", "Place", "Metrics", "Chi²", "RS", "Audit", "AI ✓", "Natural ✓"]
        )
        self.master_table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        self.master_table.itemSelectionChanged.connect(self._on_master_selection_changed)
        master_layout.addWidget(self.master_table)
        master_group.setLayout(master_layout)

        # Detail table (right)
        detail_group = QGroupBox("Per-Payload Extraction Success")
        detail_layout = QVBoxLayout()
        self.detail_label = QLabel("(Select a row to view per-payload details)")
        detail_layout.addWidget(self.detail_label)
        self.detail_table = QTableWidget()
        self.detail_table.setColumnCount(5)
        self.detail_table.setHorizontalHeaderLabels(
            ["Payload (KB)", "AI ✓", "AI %", "Natural ✓", "Natural %"]
        )
        self.detail_table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        detail_layout.addWidget(self.detail_table)
        detail_group.setLayout(detail_layout)

        tables_splitter.addWidget(master_group)
        tables_splitter.addWidget(detail_group)
        tables_splitter.setStretchFactor(0, 60)
        tables_splitter.setStretchFactor(1, 40)
        layout.addWidget(tables_splitter, 1)

        # Footer: overall stats
        self.footer_label = QLabel("Overall — AI: — | Natural: —")
        self.footer_label.setStyleSheet("font-weight: bold; padding: 5px;")
        layout.addWidget(self.footer_label)

        # Step selection
        steps_layout = QHBoxLayout()
        steps_layout.addWidget(QLabel("Steps:"))
        self.embed_check = QCheckBox("Embed")
        self.verify_check = QCheckBox("Verify")
        self.metrics_check = QCheckBox("Metrics")
        self.chi2_check = QCheckBox("Chi²")
        self.rs_check = QCheckBox("RS")
        self.audit_check = QCheckBox("Audit")
        for cb in [self.embed_check, self.verify_check, self.metrics_check,
                   self.chi2_check, self.rs_check, self.audit_check]:
            cb.setChecked(True)
            steps_layout.addWidget(cb)
        layout.addLayout(steps_layout)

        # Strategy and place selection
        sel_layout = QHBoxLayout()
        sel_layout.addWidget(QLabel("Strategies:"))
        self.strat_combo = QComboBox()
        self.strat_combo.addItems(["All"] + list(STRATEGY_TABLE.keys()))
        self.strat_combo.setCurrentText("All")
        sel_layout.addWidget(self.strat_combo)

        sel_layout.addWidget(QLabel("Places:"))
        self.place_combo = QComboBox()
        self.place_combo.addItems(["All", "indoor", "outdoor"])
        self.place_combo.setCurrentText("All")
        sel_layout.addWidget(self.place_combo)
        layout.addLayout(sel_layout)

        # Run and stop buttons
        btn_layout = QHBoxLayout()
        self.run_button = QPushButton("▶ Run / Re-run Pipeline")
        self.run_button.clicked.connect(self._on_run_clicked)
        self.stop_button = QPushButton("⏹ Stop")
        self.stop_button.setEnabled(False)
        self.stop_button.clicked.connect(self.stop_requested.emit)
        btn_layout.addWidget(self.run_button)
        btn_layout.addWidget(self.stop_button)
        btn_layout.addStretch()
        layout.addLayout(btn_layout)

        widget.setLayout(layout)
        return widget

    def _build_progress_section(self) -> QWidget:
        """Build progress bar, log, and extraction tally."""
        widget = QGroupBox("Execution Progress")
        layout = QVBoxLayout()

        self.progress_bar = QProgressBar()
        self.progress_bar.setValue(0)
        layout.addWidget(self.progress_bar)

        self.log_label = QLabel("Ready")
        self.log_label.setWordWrap(True)
        self.log_label.setMaximumHeight(60)
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setWidget(self.log_label)
        layout.addWidget(scroll)

        self.tally_label = QLabel("Extraction tallies will appear here")
        layout.addWidget(self.tally_label)

        widget.setLayout(layout)
        return widget

    def _build_result_tabs(self):
        """Build tabbed result viewers."""
        self.tab_quality_psnr = self._new_canvas_tab()
        self.tab_quality_ssim = self._new_canvas_tab()
        self.tab_quality_mse = self._new_canvas_tab()
        self.tab_chi2 = self._new_canvas_tab()
        self.tab_rs = self._new_canvas_tab()
        self.tab_pairdiff = self._new_canvas_tab()
        self.tab_rank_imp = self._new_canvas_tab()
        self.tab_rank_und = self._new_canvas_tab()

        self.tabs.addTab(self.tab_quality_psnr, "Quality (PSNR)")
        self.tabs.addTab(self.tab_quality_ssim, "Quality (SSIM)")
        self.tabs.addTab(self.tab_quality_mse, "Quality (MSE)")
        self.tabs.addTab(self.tab_chi2, "Chi-square")
        self.tabs.addTab(self.tab_rs, "RS Analysis")
        self.tabs.addTab(self.tab_pairdiff, "Pair-diff")
        self.tabs.addTab(self.tab_rank_imp, "Ranking (Imperceptibility)")
        self.tabs.addTab(self.tab_rank_und, "Ranking (Undetectability)")

        # Report tab (text + heatmaps)
        report_widget = QWidget()
        report_layout = QVBoxLayout()
        self.report_text = QTextEdit()
        self.report_text.setReadOnly(True)
        report_layout.addWidget(QLabel("Verdict:"))
        report_layout.addWidget(self.report_text, 1)
        report_widget.setLayout(report_layout)
        self.tabs.addTab(report_widget, "Report")

        # Lazy pair-diff: compute the figure only when the user opens that tab.
        self._pairdiff_tab_index = self.tabs.indexOf(self.tab_pairdiff)
        self._pairdiff_rendered = False
        self.tabs.currentChanged.connect(self._on_tab_changed)

        self.refresh_results()

    def _new_canvas_tab(self) -> QWidget:
        """Canvas fills the tab and scales with the window — fits the screen, no scroll."""
        widget = QWidget()
        layout = QVBoxLayout(widget)
        layout.setContentsMargins(0, 0, 0, 0)

        canvas = FigureCanvas(plt.Figure(figsize=(10, 7)))

        layout.addWidget(canvas, 1)
        widget.setLayout(layout)
        widget._canvas = canvas
        return widget

    def refresh_checklist(self):
        """Scan disk and update master/detail tables."""
        strategies = list(STRATEGY_TABLE.keys())
        places = ["indoor", "outdoor"]
        self.current_scan = scan_results(strategies, places)

        # Populate master table (one row per (method, place, channel) combination)
        self.master_table.setRowCount(len(self.current_scan))
        for row, scan_result in enumerate(self.current_scan):
            display = scan_result["display"]  # Shows method or method_channel
            place = scan_result["place"]
            metrics_icon = get_status_icon(scan_result["has_metrics"])
            chi2_icon = get_status_icon(scan_result["has_chi2"])
            rs_icon = get_status_icon(scan_result["has_rs"])
            audit_icon = get_status_icon(scan_result["has_audit"])

            ai_str, nat_str = format_audit_pass_rate(
                scan_result["ai_pass"], scan_result["ai_total"],
                scan_result["nat_pass"], scan_result["nat_total"]
            )

            # Add cells
            self.master_table.setItem(row, 0, QTableWidgetItem(display))
            self.master_table.setItem(row, 1, QTableWidgetItem(place))

            # Status icons
            for col, icon in enumerate([metrics_icon, chi2_icon, rs_icon, audit_icon], start=2):
                item = QTableWidgetItem(icon)
                item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
                color = QColor(0, 128, 0) if icon == "✓" else QColor(200, 0, 0)
                item.setForeground(color)
                self.master_table.setItem(row, col, item)

            # Pass rates
            self.master_table.setItem(row, 6, QTableWidgetItem(ai_str))
            self.master_table.setItem(row, 7, QTableWidgetItem(nat_str))

        # Calculate and update footer (sum all rows)
        total_ai_pass = sum(r["ai_pass"] for r in self.current_scan)
        total_ai_total = sum(r["ai_total"] for r in self.current_scan)
        total_nat_pass = sum(r["nat_pass"] for r in self.current_scan)
        total_nat_total = sum(r["nat_total"] for r in self.current_scan)

        footer_text = self._format_footer(total_ai_pass, total_ai_total, total_nat_pass, total_nat_total)
        self.footer_label.setText(footer_text)

        # Clear detail table and reset label
        self.detail_table.setRowCount(0)
        self.detail_label.setText("(Select a row to view per-payload details)")

    def _format_footer(self, ai_pass: int, ai_total: int, nat_pass: int, nat_total: int) -> str:
        """Format the footer statistics text."""
        ai_pct = (ai_pass / ai_total * 100) if ai_total > 0 else 0
        nat_pct = (nat_pass / nat_total * 100) if nat_total > 0 else 0
        ai_str = f"{ai_pass}/{ai_total} ({ai_pct:.0f}%)" if ai_total > 0 else "—"
        nat_str = f"{nat_pass}/{nat_total} ({nat_pct:.0f}%)" if nat_total > 0 else "—"
        return f"Overall — AI: {ai_str}   Natural: {nat_str}"

    def _on_master_selection_changed(self):
        """Handle master table row selection to update detail table."""
        selected = self.master_table.selectedItems()
        if not selected:
            self.detail_table.setRowCount(0)
            self.detail_label.setText("(No selection)")
            return

        # Get the row (all selected items are from the same row)
        row = self.master_table.row(selected[0])
        if row < 0 or row >= len(self.current_scan):
            return

        scan_result = self.current_scan[row]
        method = scan_result["method"]
        place = scan_result["place"]
        channel = scan_result["channel"]
        display = scan_result["display"]

        # Update detail header
        self.detail_label.setText(f"{display} — {place}: per-payload extraction success")

        # Load and populate detail data (load_audit_detail now takes channel parameter)
        detail_data = load_audit_detail(method, place, channel) if scan_result["has_audit"] else []

        self.detail_table.setRowCount(len(detail_data))
        for detail_row, payload_info in enumerate(detail_data):
            payload = payload_info["payload"]
            ai_pass = payload_info["ai_pass"]
            ai_total = payload_info["ai_total"]
            nat_pass = payload_info["nat_pass"]
            nat_total = payload_info["nat_total"]

            ai_pct = (ai_pass / ai_total * 100) if ai_total > 0 else 0
            nat_pct = (nat_pass / nat_total * 100) if nat_total > 0 else 0

            ai_str = f"{ai_pass}/{ai_total}"
            nat_str = f"{nat_pass}/{nat_total}"

            self.detail_table.setItem(detail_row, 0, QTableWidgetItem(str(payload)))
            self.detail_table.setItem(detail_row, 1, QTableWidgetItem(ai_str))
            self.detail_table.setItem(detail_row, 2, QTableWidgetItem(f"{ai_pct:.0f}%"))
            self.detail_table.setItem(detail_row, 3, QTableWidgetItem(nat_str))
            self.detail_table.setItem(detail_row, 4, QTableWidgetItem(f"{nat_pct:.0f}%"))

            # Color-code percentages: green ≥90, amber 50-89, red <50
            for col, pct in [(2, ai_pct), (4, nat_pct)]:
                item = self.detail_table.item(detail_row, col)
                if pct >= 90:
                    item.setForeground(QColor(0, 128, 0))
                elif pct >= 50:
                    item.setForeground(QColor(200, 128, 0))
                else:
                    item.setForeground(QColor(200, 0, 0))

    def refresh_results(self):
        """Reload plots and report from disk CSVs.

        Pair-diff is intentionally NOT computed here because it reads many PNGs
        from disk; it is rendered lazily the first time the user opens that tab.
        """
        strategies = list(STRATEGY_TABLE.keys())
        places = ["indoor", "outdoor"]

        # Pair-diff laziness state: reset every refresh, so re-running the pipeline
        # forces a recompute when the user re-opens the tab.
        self._pairdiff_rendered = False

        try:
            # Load aggregated data
            self.current_agg = aggregate(strategies, places)

            # Refresh CSV-based plot tabs (fast — pure DataFrame work).
            self._set_canvas_fig(self.tab_quality_psnr,
                                fig_quality_2x2(strategies, places, "PSNR"))
            self._set_canvas_fig(self.tab_quality_ssim,
                                fig_quality_2x2(strategies, places, "SSIM"))
            self._set_canvas_fig(self.tab_quality_mse,
                                fig_quality_2x2(strategies, places, "MSE"))
            self._set_canvas_fig(self.tab_chi2, fig_chi2_2x2(strategies, places))
            self._set_canvas_fig(self.tab_rs, fig_rs_2x2(strategies, places))

            # Pair-diff: show a placeholder; real figure built on first tab visit.
            self._set_canvas_fig(self.tab_pairdiff,
                                _pairdiff_placeholder_figure())

            if not self.current_agg.empty:
                self._set_canvas_fig(self.tab_rank_imp,
                                    fig_rank_heatmap(self.current_agg, "imperceptibility"))
                self._set_canvas_fig(self.tab_rank_und,
                                    fig_rank_heatmap(self.current_agg, "undetectability"))
                self.report_text.setText(build_verdict_text(self.current_agg))
            else:
                self.report_text.setText("No aggregated data available.")

        except Exception as e:
            print(f"Error refreshing results: {e}")
            self.report_text.setText(f"Error: {e}")

    def _set_canvas_fig(self, tab_widget, figure):
        """Swap in a new figure and size it to the canvas so it fits without scrolling."""
        if hasattr(tab_widget, '_canvas'):
            canvas = tab_widget._canvas
            figure.set_canvas(canvas)
            canvas.figure = figure
            dpi = figure.get_dpi()
            w = max(canvas.width(), 1)
            h = max(canvas.height(), 1)
            figure.set_size_inches(w / dpi, h / dpi, forward=False)
            canvas.draw_idle()

    def _on_run_clicked(self):
        """Handle run button click."""
        strategies = self._get_selected_strategies()
        places = self._get_selected_places()
        steps = self._get_selected_steps()

        if not strategies or not places or not steps:
            QMessageBox.warning(self, "Selection Error",
                               "Select at least one strategy, place, and step.")
            return

        # Check for re-embedding
        if "embed" in steps and self._embeddings_exist(strategies, places):
            reply = QMessageBox.question(
                self, "Embeddings Exist",
                "Embeddings already exist for selected strategies/places. Re-embed?",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            )
            if reply == QMessageBox.StandardButton.No:
                steps = [s for s in steps if s != "embed"]

        self.run_button.setEnabled(False)
        self.stop_button.setEnabled(True)
        self.run_requested.emit(strategies, places, steps)

    def _get_selected_strategies(self) -> list[str]:
        """Get selected strategies from combo."""
        sel = self.strat_combo.currentText()
        if sel == "All":
            return list(STRATEGY_TABLE.keys())
        return [sel]

    def _get_selected_places(self) -> list[str]:
        """Get selected places from combo."""
        sel = self.place_combo.currentText()
        if sel == "All":
            return ["indoor", "outdoor"]
        return [sel]

    def _get_selected_steps(self) -> list[str]:
        """Get selected steps from checkboxes."""
        steps = []
        if self.embed_check.isChecked():
            steps.append("embed")
        if self.verify_check.isChecked():
            steps.append("verify")
        if self.metrics_check.isChecked():
            steps.append("metrics")
        if self.chi2_check.isChecked():
            steps.append("chi2")
        if self.rs_check.isChecked():
            steps.append("rs")
        if self.audit_check.isChecked():
            steps.append("audit")
        return steps

    def _embeddings_exist(self, strategies: list[str], places: list[str]) -> bool:
        """Check if embeddings exist for any selected strategy/place."""
        # Check if any selected strategy/place combo has audit data (indicating embeddings were run)
        for scan_result in self.current_scan:
            if scan_result["method"] in strategies and scan_result["place"] in places:
                if scan_result["has_audit"] or scan_result["ai_total"] > 0:
                    return True
        return False

    def _on_tab_changed(self, index: int) -> None:
        """Lazily render the pair-diff figure the first time its tab is opened."""
        if index != self._pairdiff_tab_index or self._pairdiff_rendered:
            return
        self._pairdiff_rendered = True

        self._set_canvas_fig(self.tab_pairdiff,
                             _pairdiff_placeholder_figure(
                                 message="Computing pair-difference histograms..."))
        from PyQt6.QtWidgets import QApplication
        QApplication.processEvents()

        strategies = list(STRATEGY_TABLE.keys())
        places = ["indoor", "outdoor"]
        try:
            self._set_canvas_fig(self.tab_pairdiff,
                                 fig_pairdiff_2x2(strategies, places))
        except Exception as exc:  # noqa: BLE001
            self._set_canvas_fig(self.tab_pairdiff,
                                 _pairdiff_placeholder_figure(
                                     message=f"Pair-diff error: {exc}"))

    def on_progress(self, stage_label: str, current: int, total: int):
        """Update progress from worker."""
        self.progress_bar.setMaximum(total)
        self.progress_bar.setValue(current)
        self.log_label.setText(f"[{current}/{total}] {stage_label}")

    def on_verify_tally(self, strategy: str, place: str, img_type: str, tally: dict):
        """Update extraction tally from worker."""
        msg = f"{strategy} | {place} | {img_type}: {tally['pass']}✓ {tally['fail']}✗ {tally['error']}⚠"
        self.tally_label.setText(msg)

    def on_finished(self):
        """Handle worker finished."""
        self.run_button.setEnabled(True)
        self.stop_button.setEnabled(False)
        self.refresh_checklist()
        self.refresh_results()
        QMessageBox.information(self, "Done", "Batch pipeline completed!")

    def on_error(self, error_msg: str):
        """Handle worker error."""
        self.run_button.setEnabled(True)
        self.stop_button.setEnabled(False)
        QMessageBox.critical(self, "Error", f"Pipeline error:\n{error_msg}")
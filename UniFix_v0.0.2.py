# -*- coding: utf-8 -*-
"""
Created on Sun Oct  5 22:43:30 2025

@author: Frank
"""

"""
thickness_gui_auto.py

PyQt5 GUI for ThicknessDistributionCalculator.

Features:
 - Auto compute thickness map on startup and after parameter changes (0.5s debounce).
 - Profile computation runs after thickness map finishes.
 - DE optimization runs only when "Run" pressed.
 - Map, Profile, Mask shown in separate side-by-side dock widgets.
 - Colormap dropdowns for map and mask.
 - Each heavy task runs in its own QThread.

Place your ThicknessDistributionCalculator class above this file or make it importable.
"""

import sys
import traceback
import json
from typing import Dict, Any, Optional, Tuple
import numpy as np

from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QDockWidget, QTableWidget, QTableWidgetItem,
    QVBoxLayout, QPushButton, QHBoxLayout, QLabel, QDoubleSpinBox, QComboBox,
    QCheckBox, QSpinBox, QFileDialog, QMessageBox, QLineEdit, QFrame, QStyledItemDelegate
)
from PyQt5.QtGui import QBrush, QColor
from PyQt5.QtCore import Qt, QThread, pyqtSignal, QTimer

from matplotlib.figure import Figure
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar

from sputter_tdc import *


class PropertyDelegate(QStyledItemDelegate):
    def __init__(self, properties, parent=None):
        super().__init__(parent)
        self.properties = properties

    def createEditor(self, parent, option, index):
        # Holen der Property-Definition basierend auf der Zeile
        prop = self.properties[index.row()]
        
        # Erstellen des passenden Widgets
        if prop['widget_type'] == QDoubleSpinBox:
            editor = QDoubleSpinBox(parent)
            if 'value_range' in prop:
                editor.setRange(*prop['value_range'])
            if 'step_size' in prop:
                editor.setSingleStep(prop['step_size'])
            return editor
            
        elif prop['widget_type'] == QLineEdit:
            return QLineEdit(parent)
            
        return super().createEditor(parent, option, index)

    def setEditorData(self, editor, index):
        # Daten aus dem Model in den Editor laden
        value = index.data(Qt.EditRole)
        if isinstance(editor, QDoubleSpinBox):
            editor.setValue(float(value) if value else 0.0)
        elif isinstance(editor, QLineEdit):
            editor.setText(str(value) if value else "")

    def setModelData(self, editor, model, index):
        # Daten aus dem Editor ins Model schreiben
        if isinstance(editor, QDoubleSpinBox):
            model.setData(index, editor.value(), Qt.EditRole)
        elif isinstance(editor, QLineEdit):
            model.setData(index, editor.text(), Qt.EditRole)

# ---------------------- Worker Threads ---------------------- #
class MapWorker(QThread):
    """
    Worker thread that computes thickness map by calling calculate_thickness_distribution().
    Emits finished_map on success or error on exception.
    """
    finished_map = pyqtSignal(object, object, object)  # xg, yg, tmap
    error = pyqtSignal(str)

    def __init__(self, params: Dict[str, Any]):
        super().__init__()
        self.params = params
        self._interrupted = False

    def run(self):
        try:
            calc = ThicknessDistributionCalculator(
                inner_radius=self.params.get("inner_radius"),
                outer_radius=self.params.get("outer_radius"),
                opening_width_cm=self.params.get("opening_width_cm"),
                source_distance_from_center_cm=self.params.get("source_distance_from_center_cm"),
                distance_from_deposition_plane_cm=self.params.get("distance_from_deposition_plane_cm"),
                grid_resolution=int(self.params.get("grid_resolution")),
                source_length_cm=self.params.get("source_length_cm"),
                num_sources=int(self.params.get("num_sources")),
                source_spacing_cm=(None if self.params.get("source_spacing_cm") == 0 else self.params.get("source_spacing_cm")),
                source_width_cm=self.params.get("source_width_cm"),
                cos_theta0=self.params.get("cos_theta0"),
                sigma=self.params.get("sigma"),
                normalize=bool(self.params.get("normalize")),
                shift_y_cm=(None if self.params.get("shift_y_cm") == 0 else self.params.get("shift_y_cm")),
            )
            xg, yg, tmap = calc.calculate_thickness_distribution()
            if self._interrupted:
                return
            # Emit map and also calculator instance if caller needs it
            self.finished_map.emit((xg, yg, tmap), calc, None)
        except Exception as e:
            tb = traceback.format_exc()
            self.error.emit(f"{e}\n{tb}")

    def interrupt(self):
        """Flag the worker to stop early (best-effort)."""
        self._interrupted = True


class ProfileWorker(QThread):
    """
    Worker thread that computes radial profile using _thickness_along_line_from_map_numba()
    Accepts a calculator instance and thickness_map override.
    """
    finished_profile = pyqtSignal(object, object)  # (radii, tline), optional extra
    error = pyqtSignal(str)

    def __init__(self, calc_instance: ThicknessDistributionCalculator, substrate_inner_radius: float,
                 substrate_outer_radius: float, n_phi: int = 40000):
        super().__init__()
        self.calc = calc_instance
        self.substrate_inner_radius = substrate_inner_radius
        self.substrate_outer_radius = substrate_outer_radius
        self.n_phi = int(n_phi)
        self._interrupted = False

    def run(self):
        try:
            # Use the fast numba routine via class wrapper
            radii, tline = self.calc._thickness_along_line_from_map_numba(
                self.calc.thickness_map,
                self.substrate_inner_radius,
                self.substrate_outer_radius,
                n_phi=self.n_phi
            )
            if self._interrupted:
                return
            self.finished_profile.emit((radii, tline), None)
        except Exception as e:
            tb = traceback.format_exc()
            self.error.emit(f"{e}\n{tb}")

    def interrupt(self):
        self._interrupted = True


class DEWorker(QThread):
    """
    Worker thread for differential evolution optimization (expensive).
    Runs only when Run pressed.
    """
    finished_de = pyqtSignal(object, object, object)  # best_ctrl, mask_smoothed, (r_init,t_init),(r_best,t_best)
    error = pyqtSignal(str)

    def __init__(self, calc_instance: ThicknessDistributionCalculator, substrate_inner_radius: float,
                 substrate_outer_radius: float, n_cols: int, n_phi: int, maxiter: int, popsize: int):
        super().__init__()
        self.calc = calc_instance
        self.substrate_inner_radius = substrate_inner_radius
        self.substrate_outer_radius = substrate_outer_radius
        self.n_cols = int(n_cols)
        self.n_phi = int(n_phi)
        self.maxiter = int(maxiter)
        self.popsize = int(popsize)
        self._interrupted = False

    def run(self):
        try:
            best_ctrl, mask_smoothed, (r_init, t_init), (r_best, t_best) = self.calc.optimize_shadow_mask(
                substrate_inner_radius=self.substrate_inner_radius,
                substrate_outer_radius=self.substrate_outer_radius,
                n_cols=self.n_cols,
                n_phi=self.n_phi,
                maxiter=self.maxiter,
                popsize=self.popsize,
            )
            if self._interrupted:
                return
            self.finished_de.emit(best_ctrl, mask_smoothed, ((r_init, t_init), (r_best, t_best)))
        except Exception as e:
            tb = traceback.format_exc()
            self.error.emit(f"{e}\n{tb}")

    def interrupt(self):
        self._interrupted = True

# ---------------------- Plot Dock ---------------------- #
class PlotDock(QDockWidget):
    def __init__(self, title: str, cmap_choices: Optional[list] = None, parent=None):
        super().__init__(title, parent)
        self.setAllowedAreas(Qt.RightDockWidgetArea | Qt.LeftDockWidgetArea |
                             Qt.TopDockWidgetArea | Qt.BottomDockWidgetArea)
        self.container = QWidget()
        self.layout = QVBoxLayout()
        self.layout.setContentsMargins(4, 4, 4, 4)
        self.layout.setSpacing(2)

        # Matplotlib figure and canvas
        self.fig = Figure(figsize=(4, 4))
        self.canvas = FigureCanvas(self.fig)
        self.canvas.setSizePolicy(
            self.canvas.sizePolicy().Expanding,
            self.canvas.sizePolicy().Expanding
        )
        self.canvas.updateGeometry()
        # Navigation toolbar
        self.toolbar = NavigationToolbar(self.canvas, self)
        self.layout.addWidget(self.toolbar)
        
        # Controls frame (colormap selection)
        ctrl = QFrame()
        ctrl_layout = QHBoxLayout()
        ctrl_layout.setContentsMargins(0, 0, 0, 0)
        ctrl.setLayout(ctrl_layout)

        ctrl_layout.addWidget(QLabel("Colormap:"))
        self.cmap_box = QComboBox()
        if cmap_choices is None:
            cmap_choices = ["magma", "viridis", "plasma", "inferno", "gray", "cividis", "coolwarm"]
        self.cmap_box.addItems(cmap_choices)
        ctrl_layout.addWidget(self.cmap_box)
        ctrl_layout.addStretch()
        self.layout.addWidget(ctrl)
        self.layout.setSpacing(4)
        self.layout.addWidget(self.canvas)

        # Set container layout
        self.container.setLayout(self.layout)
        self.setWidget(self.container)

# ---------------------- Main Window ---------------------- #
class MainWindow(QMainWindow):
    """
    Main window that binds widgets, workers, and plotting.
    Handles debounce logic and worker coordination.
    """
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Thickness Distribution — Auto")
        self.resize(1500, 800)

        ## central placeholder
        #central = QLabel("Adjust parameters left. Auto-run updates map and profile.\nPress Run for optimization.")
        #central.setAlignment(Qt.AlignCenter)
        #self.setCentralWidget(central)

        # properties list
        self.param_groups = [
            {
                'title': "Source Configuration",
                'params': [
                    {'attr':'inner_radius', 'label':'Inner radius (cm)', 'widget_type':QDoubleSpinBox, 'range':(0.0, 10000.0), 'step':1.0, 'default':350.0},
                    {'attr':'outer_radius', 'label':'Outer radius (cm)', 'widget_type':QDoubleSpinBox, 'range':(0.1, 10000.0), 'step':1.0, 'default':380.0},
                    {'attr':'opening_width_cm', 'label':'Opening width (cm)', 'widget_type':QDoubleSpinBox, 'range':(0.1, 1000.0), 'step':0.1, 'default':20.0},
                    {'attr':'source_length_cm', 'label':'Target length (cm)', 'widget_type':QDoubleSpinBox, 'range':(0.0, 1000.0), 'step':0.1, 'default':30.0},
                    {'attr':'source_width_cm', 'label':'Target width (cm)', 'widget_type':QDoubleSpinBox, 'range':(0.01, 1000.0), 'step':0.01, 'default':1.0},
                    {'attr':'num_sources', 'label':'Number of targets', 'widget_type':QSpinBox, 'range':(1, 20), 'step':1, 'default':2},
                    {'attr':'source_spacing_cm', 'label':'Target spacing (0=auto)', 'widget_type':QDoubleSpinBox, 'range':(0.0, 1000.0), 'step':0.1, 'default':10.0},
                    {'attr':'source_distance_from_center_cm', 'label':'Target shift X (cm)', 'widget_type':QDoubleSpinBox, 'range':(-1000.0, 1000.0), 'step':0.1, 'default':0.0},
                    {'attr':'shift_y_cm', 'label':'Target Shift Y (cm)', 'widget_type':QDoubleSpinBox, 'range':(-500.0, 500.0), 'step':0.1, 'default':0.0},
                    {'attr':'distance_from_deposition_plane_cm', 'label':'Distance to Substrate (cm)', 'widget_type':QDoubleSpinBox, 'range':(0.0, 1000.0), 'step':0.1, 'default':5.0},
                    
                ]
            },
            {
                'title': "Deposition Characteristic",
                'params': [
                    {'attr':'cos_theta0', 'label':'cos(θ₀) ', 'widget_type':QDoubleSpinBox, 'range':(0.0, 5.0), 'step':0.1, 'default':1.0},
                    {'attr':'sigma', 'label':'σ (angular spread)', 'widget_type':QDoubleSpinBox, 'range':(0.001, 2.0), 'step':0.05, 'default':0.5},
                    {'attr':'normalize', 'label':'Normalize thickness', 'widget_type':QCheckBox, 'default':True},
                ]
            },
            {
                'title': "Substrate Definition",
                'params': [
                    {'attr':'substrate_inner_radius', 'label':'Inner radius (cm)', 'widget_type':QDoubleSpinBox, 'range':(0.0, 10000.0), 'step':1.0, 'default':351.0},
                    {'attr':'substrate_outer_radius', 'label':'Outer radius (cm)', 'widget_type':QDoubleSpinBox, 'range':(0.1, 10000.0), 'step':1.0, 'default':379.0},
                ]
            },
            {
                'title': "Calculation Settings",
                'params': [
                    {'attr':'grid_resolution', 'label':'Grid resolution (pixels)', 'widget_type':QSpinBox, 'range':(10, 2000), 'step':10, 'default':200},
                    {'attr':'n_phi', 'label':'Profile points (n_φ)', 'widget_type':QSpinBox, 'range':(1000, 200000), 'step':1000, 'default':80000},
                ]
            },
            {
                'title': "Optimization Parameters",
                'params': [
                    {'attr':'n_cols', 'label':'Mask columns', 'widget_type':QSpinBox, 'range':(3, 200), 'step':1, 'default':20},
                    {'attr':'maxiter', 'label':'DE iterations', 'widget_type':QSpinBox, 'range':(1, 500), 'step':1, 'default':50},
                    {'attr':'popsize', 'label':'DE population size', 'widget_type':QSpinBox, 'range':(3, 50), 'step':1, 'default':15},
                ]
            }
        ]

        self.widget_dict: Dict[str, QWidget] = {}
        self._build_left_dock()
        self._build_plot_docks()

        # Debounce timer for auto calculations
        self.debounce_timer = QTimer(self)
        self.debounce_timer.setInterval(500)
        self.debounce_timer.setSingleShot(True)
        self.debounce_timer.timeout.connect(self._start_map_worker)

        # worker management
        self.map_worker: Optional[MapWorker] = None
        self.profile_worker: Optional[ProfileWorker] = None
        self.de_worker: Optional[DEWorker] = None
        self.pending_map_run = False
        self.latest_params = {}

        # storage of latest results
        self.latest_map = None
        self.latest_calc_instance = None
        self.latest_profile = None
        self.latest_de_mask = None
        self.de_latest_profile = None

        # connect colormap changes
        self.map_dock.cmap_box.currentTextChanged.connect(self._update_map_colormap)
        self.mask_dock.cmap_box.currentTextChanged.connect(self._update_mask_colormap)

        # Auto-start on launch
        QTimer.singleShot(100, self._trigger_debounced_update)

    def _build_left_dock(self):
        dock = QDockWidget("Parameters", self)
        dock.setAllowedAreas(Qt.LeftDockWidgetArea)
        container = QWidget()
        layout = QVBoxLayout()
        layout.setContentsMargins(4, 4, 4, 4)

        self.param_table = QTableWidget(0, 2)
        self.param_table.setHorizontalHeaderLabels(["Parameter", "Control"])
        self.param_table.horizontalHeader().setStretchLastSection(True)
        # Hide row numbers (vertical header)
        self.param_table.verticalHeader().setVisible(False)
        self.param_table.setShowGrid(True)
        font = self.param_table.font()
        font.setPointSizeF(font.pointSizeF() * 1.1)
        self.param_table.setFont(font)

        # Build grouped parameter table with editable widgets
        total_rows = sum(len(g["params"]) for g in self.param_groups) + len(self.param_groups)
        self.param_table.setRowCount(total_rows)
        self.param_table.setColumnCount(2)
        self.param_table.setHorizontalHeaderLabels(["Parameter", "Value"])
        
        self.widget_dict = {}  # To store widgets for later access
        
        row_idx = 0
        for i, group in enumerate(self.param_groups):
            # Add section header row (non-editable)
            title_item = QTableWidgetItem(group['title'])
            font = title_item.font()
            font.setBold(True)
            title_item.setFont(font)
            self.param_table.setItem(row_idx, 0, title_item)
        
            # Style the header row
            for col in range(2):
                if not self.param_table.item(row_idx, col):
                    self.param_table.setItem(row_idx, col, QTableWidgetItem(""))
                item = self.param_table.item(row_idx, col)
                item.setBackground(QBrush(QColor(200, 215, 250)))  # Light blue
                if col == 1:
                    item.setText("")
        
            row_idx += 1
        
            # Add parameters for this group with actual widgets
            for param in group["params"]:
                # Column 0: Parameter label (from 'label' field)
                self.param_table.setItem(row_idx, 0, QTableWidgetItem(param['label']))
        
                # Create appropriate widget based on type and set its properties
                if param['widget_type'] == QDoubleSpinBox:
                    widget = QDoubleSpinBox()
                    if 'range' in param:
                        widget.setRange(*param['range'])
                    if 'step' in param:
                        widget.setSingleStep(param['step'])
                    if 'default' in param:
                        widget.setValue(float(param['default']))
                    widget.valueChanged.connect(self._trigger_debounced_update)
                elif param['widget_type'] == QSpinBox:
                    widget = QSpinBox()
                    if 'range' in param:
                        widget.setRange(*param['range'])
                    if 'step' in param:
                        widget.setSingleStep(param['step'])
                    if 'default' in param:
                        widget.setValue(int(param['default']))
                    widget.valueChanged.connect(self._trigger_debounced_update)
                elif param['widget_type'] == QCheckBox:
                    widget = QCheckBox()
                    if 'default' in param:
                        widget.setChecked(bool(param['default']))
                        widget.stateChanged.connect(self._trigger_debounced_update)
                elif param['widget_type'] == QComboBox:
                    widget = QComboBox()
                    if 'items' in param:
                        widget.addItems(param['items'])
                    default = param.get('default', None)
                    if default is not None:
                        if isinstance(default, bool):
                            widget.setCurrentIndex(0 if default else 1)
                    widget.currentIndexChanged.connect(self._trigger_debounced_update)                
                else:  # Default case - just a label
                    widget = QLabel(str(param.get('default', '')))
                    widget.textChanged.connect(self._trigger_debounced_update)
                    widget.setAlignment(Qt.AlignCenter)
        
                # Store the widget for later access
                self.widget_dict[param['attr']] = widget
        
                # Place widget in column 1 and set alignment for label column
                self.param_table.setCellWidget(row_idx, 1, widget)
                item0 = self.param_table.item(row_idx, 0)
                if item0:
                    item0.setTextAlignment(Qt.AlignLeft | Qt.AlignVCenter)
        
                row_idx += 1

            # Add separator row between groups (except after last group)
            if i < len(self.param_groups) - 1:
                self.param_table.insertRow(row_idx)
                for col in range(2):
                    sep_item = QTableWidgetItem("")
                    sep_item.setBackground(QBrush(QColor(230, 230, 250)))  # Light gray
                    self.param_table.setItem(row_idx, col, sep_item)
                self.param_table.setRowHeight(row_idx, 1)  # Thin separator
                row_idx += 1

        # Adjust column widths to fit content
        self.param_table.resizeColumnsToContents()

        layout.addWidget(self.param_table)

        # Buttons
        btn_row = QHBoxLayout()
        self.run_btn = QPushButton("Run (DE)")
        self.save_btn = QPushButton("Save cfg")
        self.load_btn = QPushButton("Load cfg")
        btn_row.addWidget(self.run_btn)
        btn_row.addWidget(self.save_btn)
        btn_row.addWidget(self.load_btn)
        layout.addLayout(btn_row)

        self.run_btn.clicked.connect(self.on_run_de)
        self.save_btn.clicked.connect(self.on_save)
        self.load_btn.clicked.connect(self.on_load)

        container.setLayout(layout)
        dock.setWidget(container)
        dock.setMinimumWidth(325)
        self.addDockWidget(Qt.LeftDockWidgetArea, dock)

    def _build_plot_docks(self):
        # Create docks
        self.map_dock = PlotDock("Thickness Map")
        self.profile_dock = PlotDock("Radial Profiles")
        self.mask_dock = PlotDock("Shadow Mask")

        # Add to right and split them side-by-side
        self.addDockWidget(Qt.RightDockWidgetArea, self.map_dock)
        self.addDockWidget(Qt.RightDockWidgetArea, self.profile_dock)
        self.addDockWidget(Qt.RightDockWidgetArea, self.mask_dock)

        # split horizontally so they are side-by-side
        self.splitDockWidget(self.map_dock, self.profile_dock, Qt.Horizontal)
        self.splitDockWidget(self.profile_dock, self.mask_dock, Qt.Horizontal)

        # hide the extra cmap on profile dock (keep default)
        self.map_dock.layout.itemAt(0).widget().hide()
        self.mask_dock.layout.itemAt(0).widget().hide()
        self.profile_dock.cmap_box.hide()
        self.profile_dock.layout.itemAt(1).widget().hide()

    def _collect_parameters(self) -> Dict[str, Any]:
        params: Dict[str, Any] = {}
        for group in self.param_groups:
            for param in group['params']:
                attr = param['attr']
                if attr not in self.widget_dict:
                    continue
    
                widget = self.widget_dict[attr]
                if isinstance(widget, QDoubleSpinBox):
                    params[attr] = float(widget.value())
                elif isinstance(widget, QSpinBox):
                    params[attr] = int(widget.value())
                elif isinstance(widget, QComboBox):
                    txt = widget.currentText()
                    # Handle specific cases like boolean as strings
                    if txt in ("True", "False"):
                        params[attr] = (txt == "True")
                    else:
                        params[attr] = txt
                elif isinstance(widget, QCheckBox):
                    params[attr] = bool(widget.isChecked())
                else:  # Fallback for other widget types or labels
                    try:
                        params[attr] = float(widget.text())
                    except (ValueError, AttributeError):
                        params[attr] = widget.text() if hasattr(widget, 'text') else str(widget)
        return params

    def _trigger_debounced_update(self, *args):
        """Called on parameter changes. Restart debounce timer."""
        self.debounce_timer.start()

    def _start_map_worker(self):
        """Start or queue a map worker with current parameters."""
        params = self._collect_parameters()
        self.latest_params = params

        # If map worker is running, set pending flag and return.
        if self.map_worker is not None and self.map_worker.isRunning():
            self.pending_map_run = True
            return

        # start new map worker
        self.map_worker = MapWorker(params)
        self.map_worker.finished_map.connect(self._on_map_finished)
        self.map_worker.error.connect(self._on_worker_error)
        self.map_worker.start()
        self.statusBar().showMessage("Computing thickness map...")

    def _on_map_finished(self, map_tuple, calc_instance, _):
        """Receive map, store and update map plot; then trigger profile worker."""
        xg, yg, tmap = map_tuple
        self.latest_map = (xg, yg, tmap)
        self.latest_calc_instance = calc_instance
        self._plot_map()  # update map plot immediately
        self.statusBar().showMessage("Thickness map ready", 3000)

        # start profile worker using calc instance
        try:
            params = self.latest_params
            substrate_inner = params.get("substrate_inner_radius")
            substrate_outer = params.get("substrate_outer_radius")
            n_phi = params.get("n_phi")
            # interrupt existing profile worker if running
            if self.profile_worker is not None and self.profile_worker.isRunning():
                self.profile_worker.interrupt()
            self.profile_worker = ProfileWorker(calc_instance, substrate_inner, substrate_outer, n_phi=n_phi or 40000)
            self.profile_worker.finished_profile.connect(self._on_profile_finished)
            self.profile_worker.error.connect(self._on_worker_error)
            self.profile_worker.start()
            self.statusBar().showMessage("Computing radial profile...", 2000)
        except Exception as e:
            self._on_worker_error(str(e))

        # If a pending run was requested during this run, start it now.
        if self.pending_map_run:
            self.pending_map_run = False
            # Start latest params (which might have been updated)
            QTimer.singleShot(10, self._start_map_worker)

    def _on_profile_finished(self, profile_tuple, _):
        """Receive profile and plot it."""
        radii, tline = profile_tuple
        self.latest_profile = (radii, tline)
        self._plot_profile()
        self.statusBar().showMessage("Profile ready", 3000)

    def on_run_de(self):
        """Run expensive DE optimization in its own thread. Uses latest calc instance and map."""
        if self.latest_calc_instance is None:
            QMessageBox.warning(self, "No map", "Compute thickness map first before running optimization.")
            return
        # disable run button while optimizing
        self.run_btn.setEnabled(False)
        params = self._collect_parameters()
        try:
            self.de_worker = DEWorker(
                calc_instance=self.latest_calc_instance,
                substrate_inner_radius=params.get("substrate_inner_radius"),
                substrate_outer_radius=params.get("substrate_outer_radius"),
                n_cols=params.get("n_cols"),
                n_phi=params.get("n_phi"),
                maxiter=params.get("maxiter"),
                popsize=params.get("popsize"),
            )
            self.de_worker.finished_de.connect(self._on_de_finished)
            self.de_worker.error.connect(self._on_worker_error)
            self.de_worker.start()
            self.statusBar().showMessage("Running optimization (DE)...")
        except Exception as e:
            self._on_worker_error(str(e))

    def _on_de_finished(self, best_ctrl, mask_smoothed, profiles_tuple):
        """Handle DE finished: plot mask and profiles."""
        (r_init, t_init), (r_best, t_best) = profiles_tuple
        self.latest_de_mask = mask_smoothed
        # plot mask and profiles
        self._plot_mask()
        self.de_latest_profile = (r_best, t_best)
        self._plot_profile()
        self.statusBar().showMessage("Optimization finished", 5000)
        self.run_btn.setEnabled(True)

    def _on_worker_error(self, msg: str):
        self.statusBar().clearMessage()
        QMessageBox.critical(self, "Worker Error", msg)
        self.run_btn.setEnabled(True)

    # ---------------- plotting helpers ----------------
    def _plot_map(self):
        """Plot thickness map using selected colormap and overlay source rectangles."""
        if self.latest_map is None or self.latest_calc_instance is None:
            return
        xg, yg, tmap = self.latest_map
        calc = self.latest_calc_instance
        params = self.latest_params
        cmap = self.map_dock.cmap_box.currentText()

        fig = self.map_dock.fig
        fig.clear()
        ax = fig.subplots()
        cf = ax.contourf(xg, yg, tmap, levels=20, cmap=cmap)
        fig.colorbar(cf, ax=ax, label="Normalized Thickness")
        ax.set_title("Thickness Map")
        ax.set_xlabel("Source Width [cm]")
        ax.set_ylabel("Radius [cm]")

        # --- Overlay source rectangles ---
        try:
            num_sources = int(params.get("num_sources"))
            source_width_cm = float(params.get("source_width_cm"))
            source_length_cm = float(params.get("source_length_cm"))
            inner_radius = float(params.get("inner_radius"))
            outer_radius = float(params.get("outer_radius"))
            shift_y_cm = params.get("shift_y_cm", None)
            import matplotlib.pyplot as plt

            for i in range(num_sources):
                rect_x = calc.source_x_positions[i] - source_width_cm / 2
                rect_y = (inner_radius + outer_radius) / 2 - source_length_cm / 2
                if shift_y_cm is not None and shift_y_cm != 0:
                    rect_y += calc.shift_y_positions[i]
                ax.add_patch(plt.Rectangle(
                    (rect_x, rect_y),
                    source_width_cm,
                    source_length_cm,
                    linewidth=1.5,
                    edgecolor='red',
                    facecolor='none',
                    linestyle='--'
                ))
        except Exception:
            # silently ignore if parameters missing or structure differs
            pass

        self.map_dock.canvas.draw_idle()

    def _plot_profile(self):
        """Plot radial profile in profile dock."""
        fig = self.profile_dock.fig
        fig.clear()
        ax = fig.subplots()
    
        # Plot map_worker data if available
        if hasattr(self, 'latest_profile') and self.latest_profile is not None:
            r_map, tline_map = self.latest_profile
            ax.plot(r_map, tline_map, "b-", linewidth=2, label="Source Profile")
    
        # Plot de_worker data if available
        if hasattr(self, 'de_latest_profile') and self.de_latest_profile is not None:
            r_de, tline_de = self.de_latest_profile
            ax.plot(r_de, tline_de, "r--", linewidth=2, label="Masked Profile")
    
        ax.set_xlabel("Radius [cm]")
        ax.set_ylabel("Normalized thickness")
        ax.set_title("Radial Profile")
        ax.grid(True, linestyle="--", linewidth=0.5)
    
        # Add legend only if we have at least one line
        if (hasattr(self, 'latest_profile') and self.latest_profile is not None) or \
           (hasattr(self, 'de_latest_profile') and self.de_latest_profile is not None):
            ax.legend()
    
        self.profile_dock.canvas.draw_idle()

    def _plot_mask(self):
        """Plot mask using selected colormap."""
        if self.latest_de_mask is None:
            return
        mask = np.asarray(self.latest_de_mask)
        # map extents from latest_map if available
        if self.latest_map is not None:
            xg, yg, _ = self.latest_map
            extent = [xg.min(), xg.max(), yg.min(), yg.max()]
        else:
            extent = None
        cmap = self.mask_dock.cmap_box.currentText()
        fig = self.mask_dock.fig
        fig.clear()
        ax = fig.subplots()
        im = ax.imshow(mask, origin="lower", extent=extent, interpolation="none", aspect='auto', cmap=cmap)
        fig.colorbar(im, ax=ax)
        ax.set_title("Optimized Shadow Mask")
        self.mask_dock.canvas.draw_idle()

    def _update_map_colormap(self, _):
        """Update map colormap immediately."""
        self._plot_map()

    def _update_mask_colormap(self, _):
        """Update mask colormap immediately."""
        self._plot_mask()

    # ---------------- save/load ----------------
    def on_save(self):
        params = self._collect_parameters()
        fname, _ = QFileDialog.getSaveFileName(self, "Save configuration", "", "JSON files (*.json)")
        if not fname:
            return
        with open(fname, "w") as fh:
            json.dump(params, fh, indent=2)
        self.statusBar().showMessage(f"Saved {fname}", 3000)

    def on_load(self):
        fname, _ = QFileDialog.getOpenFileName(self, "Load configuration", "", "JSON files (*.json)")
        if not fname:
            return
        with open(fname, "r") as fh:
            params = json.load(fh)
        for k, v in params.items():
            w = self.widget_dict.get(k)
            if w is None:
                continue
            try:
                if isinstance(w, QDoubleSpinBox):
                    w.setValue(float(v))
                elif isinstance(w, QSpinBox):
                    w.setValue(int(v))
                elif isinstance(w, QComboBox):
                    idx = w.findText(str(v))
                    if idx >= 0:
                        w.setCurrentIndex(idx)
                elif isinstance(w, QCheckBox):
                    w.setChecked(bool(v))
                elif isinstance(w, QLineEdit):
                    w.setText(str(v))
            except Exception:
                pass
        self.statusBar().showMessage(f"Loaded {fname}", 3000)
        # trigger an update after load
        self._trigger_debounced_update()

# ---------------------- Entrypoint ---------------------- #
def main():
    app = QApplication(sys.argv)
    w = MainWindow()
    w.show()
    sys.exit(app.exec_())

if __name__ == "__main__":
    main()

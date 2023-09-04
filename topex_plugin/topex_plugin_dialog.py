# -*- coding: utf-8 -*-
"""
/***************************************************************************
 TopexDialog
                                 A QGIS plugin
 This plugin calculates topographic exposure to wind
 Generated by Plugin Builder: http://g-sherman.github.io/Qgis-Plugin-Builder/
                             -------------------
        begin                : 2023-08-30
        git sha              : $Format:%H$
        copyright            : (C) 2023 by Manvel Khudinyan
        email                : armkhudinyan@gmail.com
 ***************************************************************************/

/***************************************************************************
 *                                                                         *
 *   This program is free software; you can redistribute it and/or modify  *
 *   it under the terms of the GNU General Public License as published by  *
 *   the Free Software Foundation; either version 2 of the License, or     *
 *   (at your option) any later version.                                   *
 *                                                                         *
 ***************************************************************************/
"""

import os
from pathlib import Path
import rasterio as rio

from qgis.PyQt import uic                   # type: ignore
from qgis.PyQt import QtWidgets             # type: ignore
from qgis.PyQt.QtCore import QSettings      # type: ignore
from qgis.PyQt.QtWidgets import QFileDialog # type: ignore
from qgis.core import (QgsProject,          # type: ignore
                       QgsRasterLayer)

import sys
sys.path.append(os.path.dirname(__file__))
from topex_plugin.topex import run_topex_analysis, get_raster_profile

# This loads your .ui file so that PyQt can populate your plugin with the elements from Qt Designer
FORM_CLASS, _ = uic.loadUiType(os.path.join(
    os.path.dirname(__file__), 'topex_plugin_dialog_base.ui'))


class TopexDialog(QtWidgets.QDialog, FORM_CLASS):
    def __init__(self, parent=None):
        """Constructor."""
        super(TopexDialog, self).__init__(parent)
        # Set up the user interface from Designer through FORM_CLASS.
        # After self.setupUi() you can access any designer object by doing
        # self.<objectname>, and you can use autoconnect slots - see
        # http://qt-project.org/doc/qt-4.8/designer-using-a-ui-file.html
        # #widgets-and-dialogs-with-auto-connect
        self.setupUi(self)

        # Load raster layer
        self.pbSelectDem.clicked.connect(self.pb_select_dem)
        self.pbSelectOutputdir.clicked.connect(self.pb_select_out_dir)
        self.pbRunTopex.clicked.connect(self.pbRunTopexAnalysis)
        self.settings = QSettings("YourPlugin", "TopexPlugin")
        self.last_selected_directory = self.settings.value("LastDirectory", "")
        # Set default values for QlineEdit
        self.leMaxDistance.setText('2000')
        self.leInterval.setText('100')
        # Populate ComboBox for dropdown topex dir selection
        self.topex_dirs = ['N', 'NE', 'E', 'SE', 'S', 'SW', 'W', 'NW', 'All']
        self.cboxTopexDirs.addItems(self.topex_dirs)

    def pbRunTopexAnalysis(self) -> None:
        self.labelStatusHolder.setText("Running...")

        # Dall parameters defined in the plugin by the user
        dem_path = Path(self.leDemDir.text())
        max_dist  = float(self.leMaxDistance.text())
        interval = float(self.leInterval.text())
        wind_dir = self.cboxTopexDirs.currentText()
        apply_mask = self.cbApplySeaMask.isChecked()
        # Run the TOPEX
        topex_result = run_topex_analysis(
            dem_path,
            wind_dir,
            max_dist,
            interval,
            apply_mask)

        # Write each TOPEX map in a separate .tif file
        profile = get_raster_profile(dem_path)
        profile.update(dtype=rio.float32, count=1)

        # TODO: check for directory to exist, otherwise create one
        out_dir = Path(self.leOutputDir.text())

        if wind_dir == 'All':
            # print(self.topex_dirs[:-1])
            for topex, name in zip(topex_result, self.topex_dirs[:-1]):
                out_name = out_dir / f'topex_{name}.tif'
                with rio.open(out_name, 'w', **profile) as dest:
                    dest.write(topex, 1)
        else:
            out_name = out_dir / f'topex_{wind_dir}.tif'
            with rio.open(out_name, 'w', **profile) as dest:
                    dest.write(topex_result, 1)
                    # self.load_raster_to_qgis_layers(out_name)

        self.labelStatusHolder.setText("Ready")


    def load_raster_to_qgis_layers(self, layer_path: Path) -> None:
        layer = QgsRasterLayer(str(layer_path), str(layer_path.name))
        QgsProject.instance().addMapLayer(layer)

        print(layer.isValid())
        # if layer.isValid():
        #     QgsProject.instance().addRasterLayer(layer)
        #     print("Raster layer loaded successfully.")
        # else:
        #     print("Error loading raster layer:", layer.dataProvider().error().message())
        QgsProject.instance().layerTreeRoot().findLayer(
            layer.id()).setItemVisibilityChecked()
        # QgsProject.instance().addMapLayer(layer)
        # iface.mapCanvas().refresh()

    def pb_select_dem(self):
        initial_dir = (self.last_selected_directory
                    if self.last_selected_directory
                    else os.path.expanduser("~"))
        dirname, _ = QFileDialog.getOpenFileName(
                    self,
                    "Select product directory",
                    initial_dir,
                    "TIFF Files (*.tif *.tiff)")
        if dirname:
            self.leDemDir.setText(dirname)
            self.last_selected_directory = str(Path(dirname).parent)
            self.settings.setValue("LastDirectory", self.last_selected_directory)

    def pb_select_out_dir(self):
        initial_dir = (self.last_selected_directory
                    if self.last_selected_directory
                    else os.path.expanduser("~"))
        dirname = QFileDialog.getExistingDirectory(
                    self,
                    "Select output directory",
                    initial_dir)
        if dirname:
            self.leOutputDir.setText(dirname)
            self.last_selected_directory = str(Path(dirname).parent)
            self.settings.setValue("LastDirectory", self.last_selected_directory)
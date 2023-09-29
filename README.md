# TOPEX
Topographic exposure, also known as TOPEX, is a measure of surrounding landforms and how they effect wind. This plugin generates a TOPEX model based on a digital elevation model (DEM).
The current wersion is limited to calculating the TOPEX for 8 directions `N, NE, E, SE, S, SW, W, NW`.

## Installation Guide
1. Clone the repository

    `git clone`

2. Create synchronized link for `topex_plugin` from GitHub repo to the QGis plugins' default directory

    - For **Linux** systems:

        Run the following command from terminal:

        `ln -s /local/path/to/Git/TOPEX/topex_plugin/ /home/user/.local/share/QGIS/QGIS3/profiles/default/python/plugins/`

    - For **Windows** systems:

        Download [*Link Shell Extention*](https://schinagl.priv.at/nt/hardlinkshellext/linkshellextension.html) and manually link the `topex_plugin` from repo to the QGis plugins default directory (possibly found at: `C:\Users\USER\AppData\Roaming\QGIS\QGIS3\profiles\default\python\plugins`)

3. In QGis *Plugins* section find *TOPEX* plugin.
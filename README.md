# TOPEX
Topographic exposure, also known as TOPEX, is a measure of surrounding landforms and how they effect wind. This plugin generates a TOPEX model based on a digital elevation model (DEM).
The current wersion is limited to calculating the TOPEX for 8 directions `N, NE, E, SE, S, SW, W, NW`.

## Installation Guide
1. Clone the repository

    `git clone git@github.com:armkhudinyan/TOPEX.git`

2. Create synchronized link for `topex_plugin` from GitHub repo to the QGis plugins' default directory

    - For **Linux** systems:

        Adapt the paths and run the following command from terminal:

        `ln -s /local/path/to/TOPEX/topex_qgis_plugin/ /home/user/.local/share/QGIS/QGIS3/profiles/default/python/plugins/`

    - For **Windows** systems:

        Download [*Link Shell Extention*](https://schinagl.priv.at/nt/hardlinkshellext/linkshellextension.html) and manually link the `topex_plugin` folder from repo to the QGis plugins default directory (possibly found at: `C:\Users\USER\AppData\Roaming\QGIS\QGIS3\profiles\default\python\plugins`)

3. In QGis *Plugins* section find *TOPEX* plugin.

> [!NOTE]
> Depending on your QGis version, you may need to manually install `rasterio` and `geopandas` third party libraries in the same env as QGis.
> In case of multiprocessing in python (not via plugn), you will need to manually install `pathos` third parthy library.

### Running from python code
Explore the source code and how to run the analysis from jupyter notebook at `TOPEX/src`.
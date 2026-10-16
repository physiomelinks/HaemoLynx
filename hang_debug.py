"""Launch napari with a hang recorder (temporary debugging aid; not committed).

Every 15 seconds this writes every thread's current Python stack to
hang_debug.log, so after napari freezes the last entry says where it is stuck.
"""
import faulthandler
from pathlib import Path

log = open(Path(__file__).with_name("hang_debug.log"), "w")
faulthandler.dump_traceback_later(15, repeat=True, file=log)

import napari

viewer = napari.Viewer()
viewer.window.add_plugin_dock_widget("HaemoLynx", "Pipeline settings")
napari.run()

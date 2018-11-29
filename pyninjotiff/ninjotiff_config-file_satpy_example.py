import os
from satpy import Scene
from datetime import datetime
from satpy.utils import debug_on
import pyninjotiff
from glob import glob
from pyresample.utils import load_area
import copy
debug_on()


chn = "IR_108"
ninjoRegion = load_area("areas.def", "nrEURO3km")

filenames = glob("data/*__")
global_scene = Scene(reader="hrit_msg", filenames=filenames)
global_scene.load([chn])
local_scene = global_scene.resample(ninjoRegion)
local_scene.save_dataset(chn, filename="msg.tif", writer='ninjotiff',
                      # ninjo product name to look for in .cfg file
                      ninjo_product_name="IR_108",
                      # custom configuration file for ninjo tiff products
                      # if not specified PPP_CONFIG_DIR is used as config file directory
                      ninjo_product_file="/config_dir/ninjotiff_products.cfg")

# we need to add a new hdf5 attribute to the old files and move them

import os
import h5py
import glob

import logging
logging.basicConfig(
    format="%(asctime)s | %(levelname)-8s | %(name)-12s | %(message)s",
    datefmt="%y-%m-%d %H:%M",
    level=logging.INFO,
)
log = logging.getLogger("script")

from tqdm.auto import tqdm
from tqdm.contrib.logging import logging_redirect_tqdm

input_path = "/scratch01.local/pspitzner/revision_1/simulations/lif/raw_alpha"
# input_path = "/Users/paul/para/2_Projects/modular_cultures/_repo/_latest/dat.nosync/simulations/lif/raw_alpha"

# get all files
files = glob.glob(input_path + "/*.hdf5")

with logging_redirect_tqdm():
    for file in tqdm(files):
        log.info(file)
        # open file
        f = h5py.File(file, "a")
        # create dset
        f.create_dataset("/meta/topology_k_in", data='~30')
        # close file
        f.close()

        # change file name on disk to include the k_in
        # stim=02_k=1_jA=45.0_jG=50.0_jM=15.0_tD=20.0_rate=80.0_stimrate=0.0_rep=000

        os.rename(file, file.replace("_jA=", "_k_in_30_jA="))

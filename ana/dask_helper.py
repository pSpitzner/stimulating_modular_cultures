# ------------------------------------------------------------------------------ #
# @Author:        F. Paul Spitzner
# @Email:         paul.spitzner@ds.mpg.de
# @Created:       2021-06-24 18:23:02
# @Last Modified: 2021-06-24 19:38:02
# ------------------------------------------------------------------------------ #

import os
import tempfile

from dask_jobqueue import SGECluster
from dask.distributed import Client, SSHCluster, LocalCluster, as_completed

client = None
cluster = None


def init_dask():
    """
        initializer for dask. never call this in the main part of your script, this
        will create a bunch of errors.

        rather, call inside functions or nest in `if __name__ == "__main__":`

        ```
        if __name__ == "__main__":
            init_dask()
        ```
    """

    global cluster
    global client

    if "sohrab" in os.uname().nodename:
        cluster = SGECluster(
            cores=1,
            memory="2GB",
            queue="rostam.q",
            death_timeout=120,
            log_directory="./log/dask/",
            local_directory="/scratch01.local/pspitzner/dask/",
            interface="ib0",
            n_workers=256,
            extra=[
                '--preload \'import sys; sys.path.append("./ana/"); sys.path.append("/home/pspitzner/code/pyhelpers/");\''
            ],
        )
    else:
        cluster = LocalCluster(local_directory=f"{tempfile.gettempdir()}/dask/")
        client = Client(cluster)




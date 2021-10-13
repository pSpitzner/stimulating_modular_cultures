# ------------------------------------------------------------------------------ #
# @Author:        F. Paul Spitzner
# @Email:         paul.spitzner@ds.mpg.de
# @Created:       2021-06-24 18:23:02
# @Last Modified: 2021-10-12 13:57:20
# ------------------------------------------------------------------------------ #

import os
import tempfile
import logging

log = logging.getLogger(__name__)

from dask_jobqueue import SGECluster
from dask.distributed import Client, SSHCluster, LocalCluster, as_completed

client = None
cluster = None


def init_dask(n_workers = 256):
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
            extra=[
                '--preload \'import sys; sys.path.append("./ana/"); sys.path.append("/home/pspitzner/code/pyhelpers/");\''
            ],
        )
        cluster.scale(cores=n_workers)

    elif "tahmineh" in os.uname().nodename:
        cluster = SGECluster(
            cores=32,
            memory="192GB",
            processes=16,
            job_extra=["-pe mvapich2-sam 32"],
            log_directory="/scratch01.local/pspitzner/dask/logs",
            local_directory="/scratch01.local/pspitzner/dask/scratch",
            interface="ib0",
            extra=[
                '--preload \'import sys; sys.path.append("./ana/"); sys.path.append("/home/pspitzner/code/pyhelpers/");\''
            ],
        )
        cluster.scale(cores=n_workers)

    elif "rudabeh" in os.uname().nodename:
        cluster = SGECluster(
            cores=32,
            memory="192GB",
            processes=16,
            job_extra=["-pe mvapich2-zal 32"],
            log_directory="/scratch01.local/pspitzner/dask/logs",
            local_directory="/scratch01.local/pspitzner/dask/scratch",
            interface="ib0",
            walltime='02:30:00',
            extra=[
                '--preload \'import sys; sys.path.append("./ana/"); sys.path.append("/home/pspitzner/code/pyhelpers/");\''
            ],
        )
        cluster.scale(cores=n_workers)

    else:
        cluster = LocalCluster(local_directory=f"{tempfile.gettempdir()}/dask/")

    client = Client(cluster)
    log.debug(cluster)
    log.debug(client)

def close():
    # cluster and client get closed automatically, when the python session terminates,
    # but sometimes we want to do it manually
    global cluster
    global client

    client.close()
    cluster.close()

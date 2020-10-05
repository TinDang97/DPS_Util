import os
import subprocess


def number_of_gpus():
    """
    Count numbers of NVIDIA GPU
    """
    return int(subprocess.getoutput("nvidia-smi --query-gpu=name --format=csv,noheader | wc -l"))


def number_of_cores():
    """
    number_of_cores()

    Detect the number of cores in this system.
    """
    # Linux, Unix and MacOS:
    if hasattr(os, "sysconf"):
        if "SC_NPROCESSORS_ONLN" in os.sysconf_names:
            # Linux & Unix:
            ncpus = os.sysconf("SC_NPROCESSORS_ONLN")
            if isinstance(ncpus, int) and ncpus > 0:
                return ncpus
        else:  # OSX:
            return int(os.popen2("sysctl -n hw.ncpu")[1].read())
    # Windows:
    if "NUMBER_OF_PROCESSORS" in os.environ:
        ncpus = int(os.environ["NUMBER_OF_PROCESSORS"])
        if ncpus > 0:
            return ncpus
    return 1  # Default


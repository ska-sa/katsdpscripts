import os
import sys
import warnings

import katdal
import katholog
import katpoint
import katsdpscripts
import scape

def git_info():
    """
    Helper function to get information about a github commit
    that was used to build the default KAT reduction packages.
    """
    return "scape: %s\nkatpoint: %s\nkatdal: %s\nkatsdpscripts: %s\nkatholog: %s"%(scape.__version__,katpoint.__version__,katdal.__version__,katsdpscripts.__version__,katholog.__version__)


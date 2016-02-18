import sys

def git_info(short=True):
    """
    Helper function to get information about a github commit
    that was used to build the default KAT reduction packages.
    """
    mods = ['katholog','katpoint','scape','katdal','katsdpscripts']
    git_info_str = ''
    if short:
        return "katsdpscripts: %s\n"%(sys.modules['katsdpscripts'].__version__)
    for m in mods:
        try:
            git_info_str += "%s: %s\n"%(sys.modules[m].__name__,sys.modules[m].__version__)
        except KeyError:
            git_info_str += "Module %s not imported\n"%m
    return git_info_str


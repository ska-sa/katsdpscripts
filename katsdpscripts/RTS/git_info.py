import sys

def git_info(mods='short'):
    """
    Helper function to get information about a github commit
    that was used to build the default KAT reduction packages.
    """
    standard_mods = ['katholog','katpoint','scape','katdal','katsdpscripts']
    git_info_str = ''

    if mods == 'short':
        try:
            git_info_str += "katsdpscripts: %s\n"%(sys.modules['katsdpscripts'].__version__)
        except KeyError:
            git_info_str += "Module katsdpscripts is not imported\n"
    elif mods == 'standard':
        for m in standard_mods:
            try:
                git_info_str += "%s: %s\n"%(sys.modules[m].__name__,sys.modules[m].__version__)
            except KeyError:
                git_info_str += "Module %s is not imported\n"%m
    elif isinstance(mods,list):
        for m in mods:
            try:
                git_info_str += "%s: %s\n"%(sys.modules[m].__name__,sys.modules[m].__version__)
            except KeyError:
                git_info_str += "Module %s is not imported\n"%m 

    return git_info_str


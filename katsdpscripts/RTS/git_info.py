import sys

def git_info(mods='short'):
    """
    Helper function to get information about a github commit
    that was used to build the default KAT reduction packages.
    
    Keywords:
    --------
    mods -- determines which modules to report.
        Valid options are 'short', 'standard' or a list of module names.

    Examples:
    --------
    git_info()
        This will return the current version of katsdpscripts.

    git_info('standard')
        This will return the versions of the five standard KAT reduction packages.

    git_info(['numpy','scape','katpoint','matplotlib'])
        This will return the versions of the packages specified in the list.

    """
    standard_mods = ['katholog','katpoint','scape','katdal','katsdpscripts']
    git_info_str = []

    if mods == 'short':
        try:
            git_info_str.append("katsdpscripts: %s"%(sys.modules['katsdpscripts'].__version__))
        except KeyError:
            git_info_str.append("Module katsdpscripts is not imported")
    elif mods == 'standard':
        for m in standard_mods:
            try:
                git_info_str.append("%s: %s"%(sys.modules[m].__name__,sys.modules[m].__version__))
            except KeyError:
                git_info_str.append("Module %s is not imported"%m)
    elif isinstance(mods,list):
        for m in mods:
            try:
                git_info_str.append("%s: %s"%(sys.modules[m].__name__,sys.modules[m].__version__))
            except KeyError:
                git_info_str.append("Module %s is not imported"%m) 

    return '\n'.join(git_info_str)


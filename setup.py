#!/usr/bin/env python
from setuptools import find_packages
from numpy.distutils.core import  setup, Extension

setup (
    name = "katsdpscripts",
    version = "trunk",
    description = "KAT observation scripting framework and SDP scripts",
    author = "MeerKAT SDP, CAM and Commissioning Teams",
    author_email = "spt@ska.ac.za",
    packages = find_packages(),
    include_package_data = True,
    scripts = [
        "RTS/Condition_Report/condition_report.py",
        "RTS/RFI_Report/rfi_report.py",
	"RTS/2.2-T_sys_T_nd/T_sys_T_nd_red.py",
        "RTS/2.8-Strong_Sources/analyse_noise_diode.py",
	"RTS/2.10-Receptor_Spectral_Baseline/analyse_spectrum.py"
    ],
    url = 'http://ska.ac.za/',
    classifiers = [
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "License :: Other/Proprietary License",
        "Operating System :: OS Independent",
        "Programming Language :: Python",
        "Topic :: Software Development :: Libraries :: Python Modules",
        "Topic :: Scientific/Engineering :: Astronomy",
    ],
    platforms = [ "OS Independent" ],
    install_requires = ['numpy', 'katpoint', 'katcp'],
    keywords = "meerkat kat ska",
    zip_safe = False,
    ext_modules = [Extension(name='gsm', sources=['RTS/gsm/gsm.f', ],
                             extra_f77_compile_args=['-std=legacy -ffixed-line-length-0'])],
    package_data = {'': ['RTS/gsm/gsm.f']}, 
)

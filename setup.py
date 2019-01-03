#!/usr/bin/env python
from setuptools import dist, find_packages

# Avoid NumPy 1.12.0 which broke f2py and cannot build gsm extension
good_numpy = 'numpy !=1.12.0b1, !=1.12.0rc1, !=1.12.0rc2, !=1.12.0'
# Ensure we have NumPy before we start as it is needed before we call setup()
# If not installed system-wide it will be downloaded into the local .eggs dir
dist.Distribution(dict(setup_requires=good_numpy))

from numpy.distutils.core import setup, Extension


setup(name="katsdpscripts",
      description="Karoo Array Telescope observation and reduction scripts",
      author="MeerKAT SDP, CAM and Commissioning Teams",
      author_email="spt@ska.ac.za",
      packages=find_packages(),
      include_package_data=True,
      scripts=[
          "RTS/Condition_Report/condition_report.py",
          "RTS/RFI_Report/rfi_report.py",
          "RTS/2.2-T_sys_T_nd/T_sys_T_nd_red.py",
          "RTS/2.8-Strong_Sources/analyse_noise_diode.py",
          "RTS/2.10-Receptor_Spectral_Baseline/analyse_spectrum.py"],
      url='https://github.com/ska-sa/katsdpscripts',
      classifiers=[
          "Development Status :: 3 - Alpha",
          "Intended Audience :: Developers",
          "License :: Other/Proprietary License",
          "Operating System :: OS Independent",
          "Programming Language :: Python",
          "Topic :: Software Development :: Libraries :: Python Modules",
          "Topic :: Scientific/Engineering :: Astronomy"],
      platforms=["OS Independent"],
      keywords="meerkat ska",
      zip_safe=False,
      ext_modules=[
          Extension(name='gsm', sources=['RTS/gsm/gsm.f'],
                    extra_f77_compile_args=['-std=legacy', '-ffixed-line-length-0'])],
      package_data={'': ['RTS/gsm/gsm.f']},
      setup_requires=['katversion', good_numpy],
      use_katversion=True,
      install_requires=['numpy', 'katpoint', 'katcp', 'scikits.fitting', 'futures'])

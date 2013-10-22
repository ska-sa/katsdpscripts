#
# Compare two sets of noise diode models.
#
# Ludwig Schwardt
# 13 September 2010
#

import numpy as np
import os.path
import matplotlib.pyplot as plt
import optparse

parser = optparse.OptionParser(usage="%prog [options] first_model_dir <second_model_dir> ...",
                               description='Plots noise diode models stored in given \
                                            directories (first in blue, second in red, etc.)')
(options, args) = parser.parse_args()
if len(args) < 1:
    parser.error("Please specify at least one directory containing noise diode models")

ants = (1, 2, 3, 4)
pols = ('H', 'V')
diodes = ('coupler', 'pin')
colors = ('b', 'r', 'g', 'k', 'c', 'm', 'y')
num_subplots = len(diodes) * len(pols)

for ant in ants:
    plt.figure(ant)
    plt.clf()
    for d, diode in enumerate(diodes):
        for p, pol in enumerate(pols):
            ind = len(pols) * d + p + 1
            plt.subplot(num_subplots, 1, ind)
            for n, nd_dir in enumerate(args):
                filename = os.path.join(nd_dir, 'T_nd_A%d%s_%s.txt' % (ant, pol, diode))
                freq, temp = np.loadtxt(filename, delimiter=',', comments='#').transpose()
                plt.plot(freq / 1e6, temp, colors[n % len(colors)])
            plt.title('Antenna %d, pol %s, %s diode' % (ant, pol, diode))
            plt.ylabel('Temp (K)')
            if ind < num_subplots:
                plt.xticks([])
            else:
                plt.xlabel('Frequency (MHz)')

plt.show()

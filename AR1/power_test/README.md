#Power test for measuring AP power consumption 

An test to move the APs in small bursts to generate maximum power consumption. Used to measure AP power consumption on M062 in the week of 18 Jan 2016.

####Observation script:
Antenna: m062 only, so that this script doesn't get run on the whole array by accident.
_If possible, no data product specified, so that no correlator or digitiser is required. Is this possible?_

Use maximum slew rate for all axes. If one axis accelerates faster and so moves the requested 20/10 degrees in less time than the other, the movement distance should be increased to have them both start and stop at the same time.

####Movement profile:
| Azimuth|Elevation|Indexer|Dwell time|
|---|---|---|---|
| 0   |15 | L | 120 s |
| 140 |25 | L | 1 s |
| 160 |35 | X | 1 s |
| 140 |25 | L | 1 s |
| 160 |35 | X | 1 s |
| 140 |25 | L | 1 s |
| 160 |35 | X | 1 s |
| 140 |25 | L | 1 s | 
| 160 |35 | X | 1 s |  
| 140 |25 | L | 1 s |
| 160 |35 | X | 1 s |
| 140 |25 | L | 60 s |
Then repeat the list from 140 degrees again.

####Reduction script:
* plot of the azimuth and elevation positions over UTC time
* plot of acceleration against time, if possible.
* also showing the antenna number and date and time of test.
* data for above plots available as a CSV or text file for plotting against power meter information
* No analysis of sky data required.


#### Example script 
power_test.py "0,15,L"   -t 120

power_test.py "140,25,L" "160,35,X" --num-repeat=5 -t 1

power_test.py "140,25,L" -t 60
    


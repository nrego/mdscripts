set buried [atomselect top "beta<1"]
set pred [atomselect top "beta=1"]

$buried set name C
$pred set name S

set buried [atomselect top "user<1"]
set pred [atomselect top "user=1"]

color Display Background green
color change rgb cyan 0.5 0.5 0.5 
color change rgb yellow 1.0 0.5 0.0


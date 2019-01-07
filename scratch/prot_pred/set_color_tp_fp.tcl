set tn [atomselect top "beta=-2"]
set tp [atomselect top "beta=1"]
set fp [atomselect top "beta=0"]
set fn [atomselect top "beta=-1"]

$tn set name C
$tp set name S
$fp set name P
$fn set name Z


color Display Background green
color change rgb cyan 0.5 0.5 0.5 
color change rgb yellow 1.0 0 0.5
color change rgb tan 0.5 0.25 0
color change rgb silver 0.25 0 0.5

color Name C cyan
color Name S yellow
color Name P tan
color Name Z silver
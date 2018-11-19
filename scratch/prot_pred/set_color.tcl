set tn [atomselect top "beta=-2"]
set tp [atomselect top "beta=1"]
set fp [atomselect top "beta=0"]
set fn [atomselect top "beta=-1"]

$tn set name H
$tp set name O
$fp set name N
$fn set name C
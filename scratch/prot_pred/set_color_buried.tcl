set buried [atomselect top "beta=-2"]
set phob [atomselect top "beta=-1"]
set phil [atomselect top "beta=0"]


$buried set name C
$buried set type C
$buried set resname UNK
$phob set name S
$phil set name P
$phil set type Z



color Display Background green
color change rgb cyan 0.5 0.5 0.5 
color change rgb silver 0.5 0.0 1.0
color Name S blue2
color Name P white

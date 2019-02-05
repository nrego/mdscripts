set phob [atomselect top "resname CH3"]
set phil [atomselect top "resname OH"]
set pred [atomselect top "beta < 0.5"]

$phob set name C
$phil set name O
$pred set name S

color Display Background white
color change rgb cyan 0.5 0.5 0.5 
color change rgb yellow 1.0 0.5 0.0

color Name C cyan
color Name O blue2
color Name S yellow

set buried [atomselect top "beta<=0"]
set contact [atomselect top "beta>0"]

$buried set name C
$contact set name S

color Display Background green
color change rgb cyan 0.5 0.5 0.5 
color change rgb yellow 0.5 0.0 1.0

color Name C cyan
color Name S yellow
set phob [atomselect top "name CH3"]
set phil [atomselect top "name OH"]
set p [atomselect top "name P"]
set n [atomselect top "name N"]

$phob set name C
$phil set name O
$p set name P
$n set name N

color Name C gray
color Name O green
color Name P blue
color Name N red
set phob [atomselect top "name CH3"]
set phil [atomselect top "name OH"]
set neg [atomselect top "name N"]
set pos [atomselect top "name P"]

$phob set name C
$phil set name O
$neg set name N
$pos set name P

color Name C gray
color Name O green
color Name N red
color Name P blue

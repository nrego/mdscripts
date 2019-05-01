draw delete all
set xmin 20
set xmax 23
set ymin 25
set ymax 80
set zmin 30
set zmax 95

draw line "$xmin $ymin $zmin" "$xmin $ymin $zmax"
draw line "$xmin $ymin $zmin" "$xmin $ymax $zmin"
draw line "$xmin $ymin $zmax" "$xmin $ymax $zmax"
draw line "$xmin $ymax $zmin" "$xmin $ymax $zmax"
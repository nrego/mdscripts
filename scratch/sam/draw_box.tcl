
proc draw_box {xmin ymin zmin xmax ymax zmax} {
    draw delete all

    draw line "$xmin $ymin $zmin" "$xmin $ymin $zmax"
    draw line "$xmin $ymin $zmin" "$xmin $ymax $zmin"
    draw line "$xmin $ymin $zmax" "$xmin $ymax $zmax"
    draw line "$xmin $ymax $zmin" "$xmin $ymax $zmax"

    draw line "$xmax $ymin $zmin" "$xmax $ymin $zmax"
    draw line "$xmax $ymin $zmin" "$xmax $ymax $zmin"
    draw line "$xmax $ymin $zmax" "$xmax $ymax $zmax"
    draw line "$xmax $ymax $zmin" "$xmax $ymax $zmax"

    draw line "$xmin $ymin $zmin" "$xmax $ymin $zmin"
    draw line "$xmin $ymin $zmax" "$xmax $ymin $zmax"
    draw line "$xmin $ymax $zmin" "$xmax $ymax $zmin"
    draw line "$xmin $ymax $zmax" "$xmax $ymax $zmax"
}

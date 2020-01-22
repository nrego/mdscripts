
proc draw_box {xmin ymin zmin xmax ymax zmax} {
    draw delete all

    draw line "$xmin $ymin $zmin" "$xmin $ymin $zmax"
    draw line "$xmin $ymin $zmin" "$xmin $ymax $zmin"
    draw line "$xmin $ymin $zmax" "$xmin $ymax $zmax"
    draw line "$xmin $ymax $zmin" "$xmin $ymax $zmax"
}

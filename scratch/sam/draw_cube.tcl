
proc draw_cube {xmin ymin zmin xmax ymax zmax} {
    draw delete all
    draw color orange
    draw materials on
    draw material Transparent

    # Bottom of cube
    draw triangle "$xmin $ymin $zmin" "$xmin $ymin $zmax" "$xmin $ymax $zmax"
    draw triangle "$xmin $ymin $zmin" "$xmin $ymax $zmin" "$xmin $ymax $zmax"

    # Top of cube
    draw triangle "$xmax $ymin $zmin" "$xmax $ymin $zmax" "$xmax $ymax $zmax"
    draw triangle "$xmax $ymin $zmin" "$xmax $ymax $zmin" "$xmax $ymax $zmax"

    # side 1
    draw triangle "$xmin $ymin $zmin" "$xmax $ymin $zmin" "$xmax $ymax $zmin"
    draw triangle "$xmin $ymin $zmin" "$xmin $ymax $zmin" "$xmax $ymax $zmin"

    # side 2
    draw triangle "$xmin $ymin $zmax" "$xmax $ymin $zmax" "$xmax $ymax $zmax"
    draw triangle "$xmin $ymin $zmax" "$xmin $ymax $zmax" "$xmax $ymax $zmax"

    # side 3
    draw triangle "$xmin $ymin $zmin" "$xmax $ymin $zmin" "$xmax $ymin $zmax"
    draw triangle "$xmin $ymin $zmin" "$xmin $ymin $zmax" "$xmax $ymin $zmax"

    # side 3
    draw triangle "$xmin $ymax $zmin" "$xmax $ymax $zmin" "$xmax $ymax $zmax"
    draw triangle "$xmin $ymax $zmin" "$xmin $ymax $zmax" "$xmax $ymax $zmax"


}


proc draw_box {xmin ymin zmin xmax ymax zmax {w 1}} {
    #draw delete all
    draw color orange

    draw line "$xmin $ymin $zmin" "$xmin $ymin $zmax" width $w
    draw line "$xmin $ymin $zmin" "$xmin $ymax $zmin" width $w
    draw line "$xmin $ymin $zmax" "$xmin $ymax $zmax" width $w
    draw line "$xmin $ymax $zmin" "$xmin $ymax $zmax" width $w

    draw line "$xmax $ymin $zmin" "$xmax $ymin $zmax" width $w
    draw line "$xmax $ymin $zmin" "$xmax $ymax $zmin" width $w
    draw line "$xmax $ymin $zmax" "$xmax $ymax $zmax" width $w
    draw line "$xmax $ymax $zmin" "$xmax $ymax $zmax" width $w

    draw line "$xmin $ymin $zmin" "$xmax $ymin $zmin" width $w
    draw line "$xmin $ymin $zmax" "$xmax $ymin $zmax" width $w
    draw line "$xmin $ymax $zmin" "$xmax $ymax $zmin" width $w
    draw line "$xmin $ymax $zmax" "$xmax $ymax $zmax" width $w
}

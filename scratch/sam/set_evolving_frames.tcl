proc animatepdbs {fileformat scriptname} {

    set start 0
    set mylist { 40 60 80 100 120 160 200 240 280 320 360 400 480 600 800 }
    set filename [format $fileformat [expr $start]]
    
    puts "Rendering initial frame in PDB sequence $filename"
    source $scriptname
    color Display Background white
    set outfile snap.[format "%03d" $start].rgb
    set outfile_mod snap.[format "%03d" $start]_labeled.png
    set labelfile labels/beta_phi_[format "%03d" $start].png
    render TachyonInternal $outfile
    exec convert -composite -gravity south $outfile $labelfile $outfile_mod
    set sel [atomselect top "all"]
    set beta [$sel get beta]
    $sel set user $beta
    
    puts "Reading PDB files..."
    foreach i $mylist {
        
        set filename [format $fileformat [expr $i]]
        
        mol addfile $filename
        source $scriptname
        color Display Background white
        set outfile snap.[format "%03d" $i].rgb
        set outfile_mod snap.[format "%03d" $i]_labeled.png
        set labelfile labels/beta_phi_[format "%03d" $i].png
        render TachyonInternal $outfile
        exec convert -composite -gravity south $outfile $labelfile $outfile_mod
    }
    exec convert -delay 20 -loop 4 snap*_labeled*.png movie.gif
    eval file delete [ glob snap*rgb ]
    #eval file delete [ glob snap*png ]
}
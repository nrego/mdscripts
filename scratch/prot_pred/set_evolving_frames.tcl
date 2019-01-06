proc animatepdbs {start inc end fileformat scriptname} {


    set filename [format $fileformat [expr $start]]
    
    puts "Rendering initial frame in PDB sequence $filename"
    source $scriptname
    color Display Background white
    set outfile snap.[format "%03d" $start].rgb
    set outfile_mod snap.[format "%03d" $start]_labeled.png
    set labelfile labels/label_[format "%03d" $start].pdf
    render TachyonInternal $outfile
    exec convert -composite -gravity south $outfile $labelfile $outfile_mod
    set sel [atomselect top "all"]
    set beta [$sel get beta]
    $sel set user $beta
    incr start $inc
    puts "Reading PDB files..."
    for {set i $start} {$i <= $end} {incr i $inc} {
        
        set filename [format $fileformat [expr $i]]
        
        mol addfile $filename
        source $scriptname
        color Display Background white
        set outfile snap.[format "%03d" $i].rgb
        set outfile_mod snap.[format "%03d" $i]_labeled.png
        set labelfile labels/label_[format "%03d" $i].pdf
        render TachyonInternal $outfile
        exec convert -composite -gravity south $outfile $labelfile $outfile_mod
    }
    exec convert -delay 20 -loop 4 snap*_labeled*.png movie.gif
    eval file delete [ glob snap*rgb ]
    #eval file delete [ glob snap*png ]
}
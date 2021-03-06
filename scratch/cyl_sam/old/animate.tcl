#

proc animate_mov {fileformat} {


    set num [molinfo top get numframes]
    

    for {set i 0} {$i < $num} {incr i 1} {
        draw delete all
        animate goto $i

        draw cylinder {28.5 35 35} {31.5 35 35} radius 20. resolution 1000 filled yes
        draw material Transparent
        display update        
        set outfile [format $fileformat [expr $i]]

        puts "Rendering $outfile"

        render snapshot $outfile

    }
    

}

#

proc animate_inter {start inc end fileformat} {


    set filename [format $fileformat [expr $start]]
    
    puts "Loading initial frame in PDB sequence $filename"
    mol load gro $filename

    for {set i $start} {$i <= $end} {incr i $inc} {
        puts "Loading frame $i"
        set filename [format $fileformat [expr $i]]
        
        animate read gro $filename
        
    }
    
}


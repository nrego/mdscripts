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
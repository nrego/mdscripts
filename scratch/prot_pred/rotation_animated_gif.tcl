proc make_rotation_animated_gif {} {
	set frame 0
	for {set i 0} {$i < 360} {incr i 10} {
		set filename snap.[format "%04d" $frame].rgb
		render TachyonInternal $filename
		incr frame
		rotate y by 10
	}
	exec convert -delay 10 -loop 4 snap.*.rgb movie.gif
	eval file delete [ glob snap*rgb ]
}

proc make_mov {} {
    set start 0
    set end 101
    set inc 1
    for {set i $start} {$i <= $end} {incr i $inc} {
        
        set filename [format "/Users/nickrego/Desktop/mov%04d.jpeg" [expr $i]]
        
        set outfile mov.[format "%04d" $i].jpg
        render TachyonInternal $outfile
        
    }
}

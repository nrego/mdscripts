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

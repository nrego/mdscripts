set mol [ atomselect 0 "all" ]

set n [ molinfo 0 get numframes ]

for {set i 0} {$i < $n} {incr i} {
    $mol frame $i
    set idx [expr $i + 1]
    mol new pdb_$i.pdb
    set sel [ atomselect $idx "all" ]
    set beta [ $sel get beta ]
    $mol set user $beta
    mol delete $idx
}
import numpy as np

pt1 = np.array([18.0, 18.3, 41.22])
pt2 = np.array([32.0, 61.7, 78.81])

xpts = np.linspace(pt1[0], pt2[0], 5)
ypts = np.linspace(pt1[1], pt2[1], 6)
zpts = np.linspace(pt1[2], pt2[2], 6)

xdiff, ydiff, zdiff = np.diff(xpts)[0], np.diff(ypts)[0], np.diff(zpts)[0]

diffs = np.array([xdiff/2.0, ydiff/2.0, zdiff/2.0])
origin = pt1 + diffs

def write_string(idx, string_data):
    string = string_data[idx]
    fname = 'image_{:02d}.dx'.format(idx)
    ctr = 0
    with open(fname, 'w') as f:
        f.write("object 1 class gridpositions counts 4 5 5\n")
        f.write("origin {:1.8e} {:1.8e} {:1.8e}\n".format(origin[0], origin[1], origin[2]))
        f.write("delta {:1.8e} {:1.8e} {:1.8e}\n".format(xdiff, 0, 0))
        f.write("delta {:1.8e} {:1.8e} {:1.8e}\n".format(0, ydiff, 0))
        f.write("delta {:1.8e} {:1.8e} {:1.8e}\n".format(0,0,zdiff))
        f.write("object 2 class gridconnections counts {} {} {}\n".format(4, 5, 5))
        f.write("object 3 class array type double rank 0 items {} data follows\n".format(100))
        for ntwid in string_data:
            f.write("{:1.8e} ".format(ntwid))
            ctr += 1
            if (ctr % 3 == 0):
                f.write("\n")

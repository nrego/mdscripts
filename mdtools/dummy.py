import MDAnalysis, mdtraj

def get_unique_rows(data):
    sorted_idx = np.lexsort(data.T)
    sorted_data = data[sorted_idx, :]
    row_mask = np.append([True], np.any(np.diff(sorted_data, axis=0),1))
    return sorted_data[row_mask]

def reload_patches():
    patch1 = MDAnalysis.Universe('patch1.pdb')
    patch2 = MDAnalysis.Universe('patch2.pdb')
    patch3 = MDAnalysis.Universe('patch3.pdb')
    return patch1, patch2, patch3, patch1.atoms.positions, patch2.atoms.positions, patch3.atoms.positions


#construct grids from a list of cuboid centers - offset is half the the resolution
def get_rects_from_grids(grid_pos):
    myrects = []

    for cntr in grid_pos:
        myrects.append(rect(cntr[0], cntr[1], cntr[2], cntr[0], cntr[1], cntr[2]))

    return myrects

# Gets *all* positions from each rect - i.e. all eight corner points for each
def extract_positions_from_rects(rects):
    the_pos = []
    for myrect in rects:
        for new_pos in myrect.return_corners():
            the_pos.append(new_pos)

    return get_unique_rows(np.array(the_pos))

# Returns only the min, max bounding points of each rect
def extract_positions_min_max_from_rects(rects):
    the_pos = []
    for myrect in rects:
        for new_pos in myrect.return_min_max():
            the_pos.append(new_pos)

    return get_unique_rows(np.array(the_pos))

def get_all_segs_by_sep(arr, sep=1.0):
    assert arr.ndim == 1
    if arr.size == 0 or arr.size == 1:
        return arr

    arr_sorted = np.sort(arr)
    patches = []

    mycurr_patch = np.array([arr_sorted[0]])

    for i in range(1, arr_sorted.size):
        prev_val = arr_sorted[i-1]
        curr_val = arr_sorted[i]

        if (curr_val - prev_val > sep):
            patches.append(mycurr_patch)
            mycurr_patch = np.array([curr_val])

        else:
            mycurr_patch = np.append(mycurr_patch, curr_val)

        prev_val = curr_val


    patches.append(mycurr_patch)

    return patches


def merge_rects_z(patch):
    xvals = np.sort(np.unique(patch[:,0]))
    yvals = np.sort(np.unique(patch[:,1]))
    zvals = np.sort(np.unique(patch[:,2]))
    rects_z = []
    for xval in xvals:
        patch_x = patch[patch[:,0] == xval]
        for yval in yvals:
            patch_y = patch_x[patch_x[:,1] == yval]
            if patch_y.size == 0:
                continue

            curr_segs = get_all_segs_by_sep(patch_y[:,2])
            for seg in curr_segs:
                zmin = seg.min()
                zmax = seg.max()
                rects_z.append(rect(xval, yval, zmin, xval, yval, zmax))

    return rects_z

def merge_rects_y(patch):
    xvals = np.sort(np.unique(patch[:,0]))
    yvals = np.sort(np.unique(patch[:,1]))
    zvals = np.sort(np.unique(patch[:,2]))
    rects_y = []
    for xval in xvals:
        patch_x = patch[patch[:,0] == xval]
        for zval in zvals:
            patch_z = patch_x[patch_x[:,2] == zval]
            if patch_z.size == 0:
                continue

            curr_segs = get_all_segs_by_sep(patch_z[:,1])
            for seg in curr_segs:
                ymin = seg.min()
                ymax = seg.max()
                rects_y.append(rect(xval, ymin, zval, xval, ymax, zval))

    return rects_y

def get_rects_with_val(rects, val, axis=0):
    return_rects = []
    for curr_rect in rects:
        if curr_rect.min_pt[axis] == val and curr_rect.max_pt[axis] == val:
            return_rects.append(curr_rect)

    return return_rects

# merge two rectangles along given axis.
#   min and max values must be the same for the rectangles
#   for every other axis
def merge_rects(rect1, rect2, axis):
    assert axis in [0,1,2]
    new_rect = rect(0,0,0,0,0,0)
    for i in range(3):
        if i != axis:
            assert rect1.min_pt[i] == rect2.min_pt[i] and rect1.max_pt[i] == rect2.max_pt[i]
            new_rect.pts[i] = rect1.min_pt[i]
            new_rect.pts[i+3] = rect1.max_pt[i]
        else:
            new_rect.pts[i] = min(rect1.min_pt[i], rect2.min_pt[i])
            new_rect.pts[i+3] = max(rect1.max_pt[i], rect2.max_pt[i])

    return new_rect



def get_unique_rows(data):
    sorted_idx = np.lexsort(data.T)
    sorted_data = data[sorted_idx, :]
    row_mask = np.append([True], np.any(np.diff(sorted_data, axis=0),1))
    return sorted_data[row_mask]

def get_atoms_to_add(positions, xmin, xmax, ymin, ymax, zmin, zmax):
    atoms_to_add = positions.copy()
    mask = (atoms_to_add[:,0] >= xmin) & (atoms_to_add[:,0] < xmax)
    atoms_to_add = atoms_to_add[mask]
    mask = (atoms_to_add[:,1] >= ymin) & (atoms_to_add[:,1] < ymax)
    atoms_to_add = atoms_to_add[mask]
    mask = (atoms_to_add[:,2] >= zmin) & (atoms_to_add[:,2] < zmax)
    atoms_to_add = atoms_to_add[mask]
    return atoms_to_add

def write_pdb(data, filename):
    top = mdtraj.Topology()
    c = top.add_chain()                                            
    for i, pos in enumerate(data):
        r = top.add_residue('RHO', c)
        a = top.add_atom('CUB', mdtraj.element.get_by_symbol('VS'), r, i)
    with mdtraj.formats.PDBTrajectoryFile(filename, 'w') as f:
        f.write(data, top)


def do_umbr_file(filename, myrects, offset=0.5):
    header_string = "; Umbrella potential for a a union of cuboid cavities\n"\
    "; Name    Type          Group  Kappa   Nstar    mu    width  cutoff  outfile    nstout\n"\
    "hydshell cuboid         OW      0.0     0       XXX    0.01   0.02   phiout.dat   50  \\\n"

    with open(filename, 'w') as fout:
        fout.write(header_string)
        for myrect in myrects:
            min_pt, max_pt = myrect.return_min_max()
            min_pt -= offset
            max_pt += offset
            xmin = min_pt[0] / 10.0
            ymin = min_pt[1] / 10.0
            zmin = min_pt[2] / 10.0
            xmax = max_pt[0] / 10.0
            ymax = max_pt[1] / 10.0
            zmax = max_pt[2] / 10.0

            outstr = "{:.3f} {:.3f} {:.3f}  {:.3f} {:.3f} {:.3f} \\\n".format(xmin, ymin, zmin, xmax, ymax, zmax)
            fout.write(outstr)


def remove_dups_from_list(mylist):
    if len(mylist) == 0 or len(mylist) == 1:
        return mylist
    else:
        new_list = []
        for val in mylist:
            if val not in new_list:
                new_list.append(val)

        return new_list

class rect:
    def __init__(self, xmin, ymin, zmin, xmax, ymax, zmax):
        self.pts = np.array([xmin, ymin, zmin, xmax, ymax, zmax], dtype=np.float64)

    def __eq__(self, other):
        return np.array_equal(self.pts, other.pts)

    @property
    def xmin(self):
        return self.pts[0]

    @property
    def xmax(self):
        return self.pts[3]   

    @property
    def ymin(self):
        return self.pts[1]    

    @property
    def ymax(self):
        return self.pts[4]    

    @property
    def zmin(self):
        return self.pts[2]   

    @property
    def zmax(self):
        return self.pts[5] 

    @property
    def min_pt(self):
        return np.array([self.xmin, self.ymin, self.zmin])

    @property
    def max_pt(self):
        return np.array([self.xmax, self.ymax, self.zmax])
    
    
    ## Returns a generator that gives all eight corners of rect
    def return_corners(self):
        for xval in [self.xmin, self.xmax]:
            for yval in [self.ymin, self.ymax]:
                for zval in [self.zmin, self.zmax]:
                    yield np.array([xval, yval, zval])

    # Returns the two defining points of the rectangle:
    #  (xmin, ymin, zmin) and (xmax, ymax, zmax)
    def return_min_max(self):
        return [self.min_pt, self.max_pt]
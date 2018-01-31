
lmbdas = np.arange(0,1,0.1)

start = 2000
dl = 0.025
lim = 200

for dirname in dirs:
    dr.loadXVG("{}/dhdl.xvg".format(dirname))
    dr.loadXVG("../rev_vdw/{}/dhdl.xvg".format(dirname))

for lmbda in lmbdas:
    lmbda = np.around(lmbda, decimals=1)
    next_lmbda = np.around(lmbda+0.1, decimals=1)

    ds_forw_0 = dr.datasets["lambda_{:.1f}/dhdl.xvg".format(lmbda)]
    ds_revr_0 = dr.datasets["../rev_vdw/lambda_{:.1f}/dhdl.xvg".format(lmbda)]

    ds_forw_1 = dr.datasets["lambda_{:.1f}/dhdl.xvg".format(next_lmbda)]
    ds_revr_1 = dr.datasets["../rev_vdw/lambda_{:.1f}/dhdl.xvg".format(next_lmbda)]

    dat_forw_0 = np.array(ds_forw_0.data[start:][next_lmbda])
    dat_revr_0 = np.array(ds_revr_0.data[start:][next_lmbda])
    dat_forw_1 = -np.array(ds_forw_1.data[start:][lmbda])
    dat_revr_1 = -np.array(ds_revr_1.data[start:][lmbda])

    min_val = np.max((np.min((dat_forw_0, dat_revr_0, dat_forw_1, dat_revr_1)), -lim))
    max_val = np.min((np.max((dat_forw_0, dat_revr_0, dat_forw_1, dat_revr_1)), lim))

    bins = np.arange(min_val, max_val+(2*dl), dl)

    hist_forw_0, bins = np.histogram(dat_forw_0, bins=bins, normed=True)
    hist_forw_1, bins = np.histogram(dat_forw_1, bins=bins, normed=True)
    hist_revr_0, bins = np.histogram(dat_revr_0, bins=bins, normed=True)
    hist_revr_1, bins = np.histogram(dat_revr_1, bins=bins, normed=True)

    lam_0_entrop = (hist_forw_0 * np.ma.log(hist_forw_0/hist_revr_0)).sum() * dl
    lam_1_entrop = (hist_forw_1 * np.ma.log(hist_forw_1/hist_revr_1)).sum() * dl

    print(r"$\lambda={:.1f};  S={:.3f}".format(lmbda, lam_0_entrop))
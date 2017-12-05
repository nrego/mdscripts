
shelltypes=[r'$R=3.0 \; \AA$', r'$R=3.5 \; \AA', r'$R=4.0 \; \AA', r'$R=4.5 \; \AA', r'$R=5.0 \; \AA', r'$R=5.5 \; \AA', r'$R=6.0 \; \AA', 'excluded volume', 'VdW radius']

for i,dirname in enumerate(dirnames):
    print("doing {}".format(dirname))
    lam_00 = np.loadtxt('{}/lambda_{:02d}/dhdl.xvg'.format(dirname, 0), comments=['@', '#'])
    lam_05 = np.loadtxt('{}/lambda_{:02d}/dhdl.xvg'.format(dirname, 5), comments=['@', '#'])
    lam_10 = np.loadtxt('{}/lambda_{:02d}/dhdl.xvg'.format(dirname, 10), comments=['@', '#'])

    lam_00 = lam_00[50000:]
    lam_05 = lam_05[50000:]
    lam_10 = lam_10[50000:]

    avg_du_00 = lam_00[:,1].mean()
    avg_du_05 = lam_05[:,1].mean()
    avg_du_10 = lam_10[:,1].mean()

    min_val = min(lam_00[:,-1].min(), lam_10[:,-3].min())
    max_val = max(lam_00[:,-1].max(), lam_10[:,-3].max())
    bins = np.arange(min_val, max_val, 0.1)
    hist_00, bb0 = np.histogram(lam_00[:,-1], bins=bins, normed=True)
    hist_10, bb1 = np.histogram(-lam_10[:,-3], bins=bins, normed=True)

    loghist_00 = np.log(hist_00)
    #loghist_00 -= loghist_00.min()

    loghist_10 = np.log(hist_10)
    #loghist_10 -= loghist_10.min()

    bc0 = np.diff(bb0) + bb0[:-1]
    bc1 = np.diff(bb1) + bb1[:-1]

    plt.plot(bc0, loghist_00, label=r'$P_0 (\Delta U)$')
    plt.plot(bc1, loghist_10, label=r'$P_1 (\Delta U)$')
    plt.xlabel(r'$\Delta U$')
    plt.ylabel(r'$\ln {P_{\lambda} (\Delta U)}$')

    plt.legend()
    plt.show()

    plt.plot([0.0, 0.5, 1.0], [avg_du_00, avg_du_05, avg_du_10], '-o')
    plt.xlabel(r'$\lambda$')
    plt.ylabel(r'$\langle \Delta U \rangle_{\lambda}$')

    plt.show()

import matplotlib as mpl
from matplotlib import rc 
import os, glob

from matplotlib import pyplot as plt

homedir = os.environ['HOME']

mpl.rcParams.update({'axes.labelsize': 50})
mpl.rcParams.update({'xtick.labelsize': 30})
mpl.rcParams.update({'ytick.labelsize': 30})
mpl.rcParams.update({'axes.titlesize':40})
mpl.rcParams.update({'legend.fontsize':15})


dat_opt = np.loadtxt('{}/Desktop/protein_prediction_summary_phi_opt.dat'.format(homedir), usecols=range(1,9))
labels = np.loadtxt('{}/Desktop/protein_prediction_summary_phi_opt.dat'.format(homedir), usecols=0, dtype=str)
row_labels = np.array([' '.join(label.split('_')) for label in labels])
dat_star = np.loadtxt('{}/Desktop/protein_prediction_summary_phi_star.dat'.format(homedir), usecols=range(1,9))

col_labels_opt = (r'$\beta \phi_\mathrm{opt}$', 'TPR', 'FPR', 'PPV', r'$d_\mathrm{h}$', r'$f_1$', 'MCC')
col_labels_star = (r'$\beta \phi^*$', 'TPR', 'FPR', 'PPV', r'$d_\mathrm{h}$', r'$f_1$', 'MCC')

row_labels = np.array(['Thymidylate Synthase', 'Mannose-Binding Protein', 'Phospholipase A2', 'MDM2', 'Ubiquitin'])

def plot_data_table(dat, row_labels, col_labels):
    ax = plt.gca()
    cell_text = []
    for row in range(row_labels.size):
        row_dat = dat[row]
        beta_phi, tp, fp, tn, fn, tpr, fpr, d_h = row_dat
        tpr = tp/(tp+fn)
        fpr = fp/(fp+tn)
        tnr = 1-fpr
        prec = tp/(tp+fp)
        d_h = 2/((1/tpr)+(1/tnr))
        f_1 = 2/((1/tpr)+(1/prec))
        mcc = ((tp*tn)-(fp*fn))/ np.sqrt((tp+fn)*(tp+fp)*(tn+fp)*(tn+fn))
        this_dat = [beta_phi, tpr, fpr, prec, d_h, f_1, mcc]
        cell_text.append('{:0.4f}  {:0.4f}  {:0.4f}  {:0.4f}  {:0.4f}  {:0.4f}  {:0.4f}'.format(*this_dat).split())
    ax.axis('tight')
    ax.axis('off')
    ax.table(cellText=cell_text,
             rowLabels=row_labels,
             colLabels=col_labels,
             loc='center')

fig, ax = plt.subplots(figsize=(6,5))
plot_data_table(dat_opt, row_labels, col_labels_opt)

fig.savefig('{}/Desktop/table_opt.pdf'.format(homedir), transparent=True, bbox_inches='tight')

fig, ax = plt.subplots()
plot_data_table(dat_star, row_labels, col_labels_star)

fig.savefig('{}/Desktop/table_star.pdf'.format(homedir), transparent=True, bbox_inches='tight')

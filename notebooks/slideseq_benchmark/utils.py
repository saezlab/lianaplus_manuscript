import pandas as pd
import numpy as np
from scipy.sparse import csr_matrix
from decoupler import p_adjust_fdr
from anndata import AnnData
from sklearn.metrics import (roc_curve, 
                             roc_auc_score,
                             precision_recall_curve,
                             auc,
                             f1_score,
                             balanced_accuracy_score,
                             precision_score,
                             recall_score)

from matplotlib import pyplot as plt
import seaborn as sns
import matplotlib.patches as mpatches

def plot_method_comparison(res, metric, baseline=None, ymin=0, ymax=1):
    # Set the figure size
    plt.figure(figsize=(7, 6))

    # Create a color palette
    palette = sns.color_palette("Set1", n_colors=res['Method'].nunique())

    # Plot each 'Method' group with a different color
    for i, method in enumerate(res['Method'].unique()):
        subset = res[res['Method'] == method]
        sns.boxplot(x='Score', y=metric, data=subset, color=palette[i], boxprops=dict(alpha=.9))

    # Plot the outlines without filling color
    sns.boxplot(x='Score', y=metric, data=res, showcaps=False, boxprops=dict(facecolor='None'),
                showfliers=False, whiskerprops=dict(color='None'))

    # Set the y-axis labels to two decimals
    plt.gca().set_yticklabels(['{:.2f}'.format(x) for x in plt.gca().get_yticks()])

    # Draw a horizontal line at y=0.50
    if baseline is not None:
        plt.axhline(y=baseline, color='red', linestyle='dashed', linewidth=1.5)

    # Rotate the x-axis labels
    plt.xticks(rotation=90)

    # Remove the title and subplot title
    plt.title('')
    plt.suptitle('')

    method_names = res['Method'].unique()
    colors = plt.cm.Set1(range(len(method_names)))

    # Create a list of patches for the legend
    legend_patches = [mpatches.Patch(color=colors[i], label=method) for i, method in enumerate(method_names)]
    plt.legend(handles=legend_patches, title='Method', bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
    
    # set y axis limits
    plt.ylim(ymin=0, ymax=1)

    # Tight layout often improves the spacing between subplots
    plt.tight_layout()

    # Show the plot
    plt.show()

def calc_weighted_f1(gt, score_key):
    y_true, y_scores = gt['truth'], gt[score_key]

    weighted_f1 = f1_score(y_true, y_scores, average='weighted')
    
    return weighted_f1


def odds_ratio(gt, score_key, top_prop=0.05):
    # Join benchmark (assumed truth) and ccc results
    # Get /w ccc_target and a response [0, 1] column
    gt = gt.sort_values(score_key, ascending=False)
    top_n = int(gt.shape[0] * top_prop)
    # assign the top rank interactions to 1
    a = np.zeros(gt.shape[0])
    a[0:top_n] = 1
    gt.loc[:, ["top_n"]] = a

    top = gt[gt["top_n"] == 1]
    tp = np.sum(top['truth'] == 1)
    fp = np.sum(top['truth']== 0)

    bot = gt[gt["top_n"] == 0]
    fn = np.sum(bot['truth'] == 1)
    tn = np.sum(bot['truth'] == 0)

    numerator = tp * tn
    denominator = fp * fn
    if denominator == 0:
        if numerator == 0:
            # undefined
            return np.nan
        else:
            # perfect score
            oddsratio = np.inf
    else:
        oddsratio = numerator / denominator
        
    return oddsratio

def onehot_groupby(adata, groupby='cell_type'):
    cts = pd.DataFrame(pd.get_dummies(adata.obs[groupby]).values.astype(np.float32),
                   index = adata.obs.index,
                   columns = pd.get_dummies(adata.obs[groupby]).columns
                   )
    ctdata = AnnData(X=csr_matrix(cts.values), obs=adata.obs, var=pd.DataFrame(index=cts.columns))
    ctdata.obsm = adata.obsm
    ctdata.obsp = adata.obsp
    
    return ctdata
    
    
def join_pred_truth(lr_res, lrdata, cpdata, lr_cols=['ligand', 'receptor'], ct_cols=['source', 'target']):
    lr_res = lr_res.copy()
    lr_truth = lrdata.var
    lr_truth['lr_truth'] = ((p_adjust_fdr((lr_truth['morans_pvals']) <= 0.05)) * (lr_truth['morans_r'] > 0)).astype(np.int8)
    lr_truth = lr_truth[lr_cols + ['lr_truth']]

    ct_truth = cpdata.var.copy().rename(columns={'ligand': 'source', 'receptor': 'target'})
    ct_truth['ct_truth'] = (p_adjust_fdr((ct_truth['morans_pvals']) <= 0.05) * (ct_truth['morans_r'] > 0)).astype(np.int8)
    ct_truth = ct_truth[ct_cols + ['ct_truth']]

    gt = lr_res.merge(lr_truth, left_on=['ligand_complex', 'receptor_complex'], right_on=lr_cols, how='inner')
    gt = gt.merge(ct_truth, left_on=ct_cols, right_on=ct_cols, how='inner')

    gt['truth'] = gt['lr_truth'] * gt['ct_truth']
    
    return gt


def generate_random_baseline(gt, score_key, metric_fun, n_perms=100, **kwargs):
    rng = np.random.default_rng()
    random_scores = []
    gt = gt.copy()
    for i in range(n_perms):
        gt[score_key] = rng.permutation(gt[score_key].values)
        random_scores.append(metric_fun(gt, score_key, **kwargs))
        
    return np.mean(random_scores)



def plot_roc(fpr, tpr, auroc, score_key=''):
    # Plot ROC curve
    plt.figure()
    plt.plot(fpr, tpr, label='ROC curve (area = %0.2f)' % auroc)
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(score_key)
    plt.legend(loc="lower right")
    plt.show()

def calc_auroc(gt, score_key, show_plot=True):
    y_true, y_pred = gt['truth'], gt[score_key]
    
    # Calculate ROC Curve and AUROC
    fpr, tpr, thresholds = roc_curve(y_true, y_pred)
    auroc = roc_auc_score(y_true, y_pred)
    if show_plot:
        plot_roc(fpr, tpr, auroc, score_key=score_key)

    return auroc

def calc_auprc(gt, score_key, show_plot=True):
    y_true, y_scores = gt['truth'], gt[score_key]

    # Calculate Precision-Recall Curve and AUPRC
    precision, recall, _ = precision_recall_curve(y_true, y_scores)
    auprc = auc(recall, precision)
    
    if show_plot:
        plot_precision_recall(recall, precision, auprc, score_key=score_key)

    return auprc

def plot_precision_recall(recall, precision, auprc, score_key=''):
    import matplotlib.pyplot as plt

    plt.figure()
    plt.plot(recall, precision, label=f'Precision-Recall curve (area = {auprc:.2f})')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title(f'Precision-Recall Curve for {score_key}')
    plt.legend(loc="lower left")
    plt.show()
    
    
def calc_weighted_f1(gt, score_key, average='binary'):
    y_true, y_scores = gt['truth'], gt[score_key]

    weighted_f1 = f1_score(y_true, y_scores, average=average)
    
    return weighted_f1

def calc_accuracy(gt, score_key):
    y_true, y_scores = gt['truth'], gt[score_key]
    accuracy = balanced_accuracy_score(y_true, y_scores, adjusted=False)
    return accuracy

def calc_precision(gt, score_key):
    y_true, y_scores = gt['truth'], gt[score_key]
    precision = precision_score(y_true, y_scores)
    return precision

def calc_recall(gt, score_key):
    y_true, y_scores = gt['truth'], gt[score_key]
    recall = recall_score(y_true, y_scores)
    return recall

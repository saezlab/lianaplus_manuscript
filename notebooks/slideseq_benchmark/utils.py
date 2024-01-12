import pandas as pd
import numpy as np
from scipy.sparse import csr_matrix
from decoupler import p_adjust_fdr
from anndata import AnnData
from sklearn.metrics import roc_curve, roc_auc_score, precision_recall_curve, auc
from matplotlib import pyplot as plt

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

    lr_truth = lrdata.var
    lr_truth['lr_truth'] = ((p_adjust_fdr((lr_truth['morans_pvals']) <= 0.05)) * (lr_truth['morans_r'] > 0)).astype(np.int8)
    lr_truth = lr_truth[lr_cols + ['lr_truth']]

    ct_truth = cpdata.var.copy().rename(columns={'ligand': 'source', 'receptor': 'target'})
    ct_truth['ct_truth'] = (p_adjust_fdr((ct_truth['morans_pvals']) <= 0.05) * (ct_truth['morans_r'] > 0)).astype(np.int8)
    ct_truth = ct_truth[ct_cols + ['ct_truth']]

    gt = lr_res.merge(lr_truth, left_on=['ligand_complex', 'receptor_complex'], right_on=lr_cols, how='inner')
    gt = gt.merge(ct_truth, left_on=ct_cols, right_on=ct_cols, how='inner')

    gt['truth'] = gt['lr_truth'] * gt['ct_truth']
    print(f"ratio: {gt['truth'].sum() / gt.shape[0]}, shape:{gt.shape}")
    
    return gt

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

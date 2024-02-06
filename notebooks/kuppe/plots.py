
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.colors import TwoSlopeNorm
from adjustText import adjust_text
import matplotlib.patches as mpatches

def create_heatmap(res, 
                   fill = 'estimate', 
                   index_key="factor",
                   column_key='pathway',
                   label='pval',
                   xlabel = None,
                   cbar_label=None,
                   max_value=None, 
                   significance_level=0.05,
                   filter_significant=True,
                   figsize=(4, 3),
                   cmap='RdBu_r',
                   ax=None,
                   group_annotations=None   
                   ):
    res = res.copy()
    if max_value is not None:
        res.loc[res[fill] > max_value, fill] = max_value
        res.loc[res[fill] < -max_value, fill] = -max_value
    
    res['significant'] = res[label].apply(lambda x: '*' if x < significance_level else '')
    
    if filter_significant:
        sig_paths = res[res[label] <= significance_level][column_key].unique()
    else:
        sig_paths = res[column_key].unique()

    # Filter for significant pathways
    filtered_data = res[res[column_key].isin(sig_paths)]
    # Create pivot table for heatmap
    heatmap_data = filtered_data.pivot(index=index_key, columns=column_key, values=fill)
    heatmap_data.sort_index(inplace=True, ascending=False)
    
    if max_value is None:
        norm = TwoSlopeNorm(vmin=-max_value, vcenter=0, vmax=max_value)
    else:
        norm = None
    
    # Plotting
    if ax is None:
        plt.figure(figsize=figsize)
        ax = plt.gca()

    heatmap = ax.pcolormesh(heatmap_data, cmap=cmap, norm=norm)

    # Adding Text Annotations
    for y, row in enumerate(heatmap_data.index):
        for x, column in enumerate(heatmap_data.columns):
            value = filtered_data.loc[(filtered_data[index_key] == row) &
                                       (filtered_data[column_key] == column), 'significant'].values[0]
            
            ax.text(x + 0.5, y + 0.4,
                     value,
                     horizontalalignment='center', 
                     verticalalignment='center',
                     color='white',
                     fontstyle='oblique',
                     fontsize=30
                     )

    if group_annotations is not None:
        for i, (row, color) in enumerate(group_annotations.items()):
            ax.add_patch(mpatches.Rectangle((i, 0), 1, 1, color=color))

        # Add a legend for the color annotations
        legend_handles = [mpatches.Patch(color=color, label=row) for row, color in group_annotations.items()]
        ax.legend(handles=legend_handles, loc='upper center', bbox_to_anchor=(0.5, -0.05), ncol=len(group_annotations))

    # Styling
    cbar = plt.colorbar(heatmap, ax=ax)
    ax.set_xticks(np.arange(0.5, len(heatmap_data.columns), 1))
    ax.set_xticklabels(heatmap_data.columns, rotation=90, fontsize=14)
    ax.set_yticks(np.arange(0.5, len(heatmap_data.index), 1))
    ax.set_yticklabels(heatmap_data.index, fontsize=14)
    cbar.ax.tick_params(labelsize=14)
    
    if xlabel is not None:
        ax.set_xlabel(xlabel, fontsize=15)
    if cbar_label is not None:
        cbar.set_label(cbar_label, size=15)
    
    ax.set_ylabel('')
    plt.tight_layout()
    plt.show()


def plot_lr_pairs(H, fct, label_fun, ax=None, figsize=(6, 6), method='average',
                  adjust_text_kwargs={'arrowprops': dict(arrowstyle='->', color='darkred')}):
    lr_pairs = H[fct].reset_index()
    lr_pairs['rank'] = lr_pairs[fct].rank(ascending=False, method=method)
    lr_pairs['name'] = lr_pairs.apply(label_fun, axis=1)

    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    ax.scatter(lr_pairs['rank'], lr_pairs[fct], s=30, c='black')

    y_max = lr_pairs[fct].max()
    ax.set_ylim(lr_pairs[fct].min(), y_max + y_max * 0.1)
    x_max = lr_pairs['rank'].max()
    ax.set_xlim(lr_pairs['rank'].min()-1, x_max + x_max * 0.05)

    ax.set_xlabel('Rank', fontsize=19)
    ax.set_ylabel(f'{fct} loadings', fontsize=19)
    ax.tick_params(axis='both', which='major', labelsize=15)
    
    texts = []
    for i, row in lr_pairs.iterrows():
        if row['name']:
            texts.append(ax.text(row['rank'], row[fct], row['name'], fontsize=13.5, color='darkred')) 

    adjust_text(texts, ax=ax, **adjust_text_kwargs)
    
    plt.tight_layout()
    

def importances_heatmap(df, target_col, predictor_col, importances_col, label_col, figsize=(6, 5.5), ax=None):
    # Pivot the DataFrame for 'importances' to create the heatmap
    pivot_table_importances = df.pivot(target_col, predictor_col, importances_col)

    # Pivot for 'label' to use for annotations
    pivot_table_labels = df.pivot(target_col, predictor_col, label_col)
    pivot_table_labels.fillna('', inplace=True)

    # Create the heatmap
    if ax is None:
        _, ax = plt.subplots(figsize=figsize)
    cmap = sns.diverging_palette(240, 10, n=9, as_cmap=True)  # Blue to red through white
    ax = sns.heatmap(pivot_table_importances, cmap=cmap, annot=False, fmt="", annot_kws={'size': 12, 'weight': 'bold'},
                     center=0, vmin=-5, vmax=5, cbar_kws={'label': 'Importances'}, ax=ax)
    cbar = ax.collections[0].colorbar
    cbar.ax.set_ylabel('Importances', size=20)
    cbar.ax.tick_params(labelsize=15)
    

    # Annotate each cell with the corresponding label
    for i, row in enumerate(pivot_table_importances.values):
        for j, val in enumerate(row):
            label = pivot_table_labels.iloc[i, j]
            ax.text(j+0.5, i+0.5, label, ha="center", va="center", fontweight="bold", size=13)

    # Customizing the plot
    ax.set_xlabel('Predictor', size=19)
    ax.set_ylabel('Target', size=19)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=90, size=15)
    ax.set_yticklabels(ax.get_yticklabels(), size=15)
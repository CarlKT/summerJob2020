# -*- coding: utf-8 -*-
"""
Examples for building graphs.

"""
import os
import pandas as pd
import numpy as np
import scipy.stats as sci
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt 
import seaborn as sns

columns = ['WT r1', 'WT r2', 'mdx r1', 'mdx r2', 'mdx TLR4-/- r1', 'mdx TLR4-/- r2']
outpath = '/Users/orsi/Desktop/'
sample_order = ['C57Bl6_r1', 'C57Bl6_r2', 'Dmdmdx_r1', 'Dmdmdx_r2', 'Bl6Bl10_r1', 'Bl6Bl10_r2']
hues = ['#2113eb', '#2113eb', '#eb1313', '#eb1313', '#3de03d', '#3de03d']
colours = ['#2113eb', '#eb1313', '#3de03d']

           
def read_PFs(folder, common_col):
    df = pd.DataFrame()
    for filename in os.listdir(folder):
        print(filename)
        temp = pd.read_csv(folder + filename, sep = '\t', skiprows = 37)
        df[filename] = pd.Series(temp.iloc[:, common_col])
    return df;


def read_proportions(folder):
    df = pd.DataFrame()
    for filename in os.listdir(folder):
        temp = pd.read_csv(folder + filename, sep = '\t')
        data = pd.Series(temp.iloc[:,7])
        split_data = data.str.split(' \(', n = 1, expand = True)
        shortened = split_data.iloc[:,0]
        region_counts = shortened.value_counts(normalize = False)
        df[filename] = region_counts
    return df;


def shorten(old_name):
    short = old_name.split('/')[-1]
    no_ext = short.split('.')[0]
    tokens = no_ext.split('_')
    sample = '_'.join([tokens[0], tokens[4]])
    return sample;


def rename_cols(df):
    for col in df.columns:
        short = shorten(col)
        df.rename(columns = { col: short }, inplace = True)
    df.rename(columns = {'C57Bl6_r1': 'WT r1', 'C57Bl6_r2': 'WT r2',
                         'Bl6Bl10_r1': 'mdx TLR4-/- r1', 'Bl6Bl10_r2': 'mdx TLR4-/- r2',
                         'Dmdmdx_r1': 'mdx r1', 'Dmdmdx_r2': 'mdx r2'}, inplace = True)
    return df;


def violin_plot(df, title):
    fig, ax = plt.subplots(figsize = (8, 6))
    ax = sns.violinplot(data = df, order = columns, palette = hues)
    ax.set_xticklabels(ax.get_xticklabels(), rotation = 45, horizontalalignment = 'right')
    ax.set_title('H3K27me3 Enriched Inflammatory Pathway Genes')
    ax.set_ylabel('tag count')
    for n in range(len(hues)):
        ax.collections[2*n].set_edgecolor(hues[n])
        ax.collections[2*n].set_facecolor('white')
    plt.show()
    fig.savefig(outpath + title + '.png', bbox_inches = "tight", dpi = 600)


def swarm_plot(df, title):
    fig, ax = plt.subplots(figsize = (8, 6))
    ax = sns.swarmplot(data = df, order = columns, palette = hues, size = 5)
    ax.set_xticklabels(ax.get_xticklabels(), rotation = 45, horizontalalignment = 'right')
    ax.set_title('OxPhos Pathway Genes Near H3K27me3 Peaks')
    ax.set_ylabel('tag count')
    plt.show()
    fig.savefig(outpath + title + '.png', bbox_inches = "tight", dpi = 600)


def heatmap(df, title, scalebar = 'log 10 tag count'):
    fig, ax = plt.subplots(figsize = (10, 10))
    ax = sns.heatmap(df, yticklabels = True, linewidth = 1,
                        cbar_kws = {'label': scalebar}, cmap = 'coolwarm')
    bottom, top = ax.get_ylim()
    ax.set_ylim(bottom + 0.5, top - 0.5)
    plt.yticks(rotation = 0, size = 12) 
    plt.xticks(rotation = 45, ha = 'right', size = 12)
    ax.figure.axes[-1].yaxis.label.set_size(12)
    plt.show()
    ax.savefig(outpath + title + '.png', bbox_inches = "tight", dpi = 600)
    
    
def heatmap_large(df, title, scalebar = 'log 10 tag count'):
    ax = sns.clustermap(df, figsize = (6, 8), yticklabels = False, method = 'ward',
                        cbar_kws = {'label': scalebar})
    plt.yticks(rotation = 0, size = 12) 
    plt.xticks(rotation = 45, ha = 'right', size = 12)
    plt.show()
    ax.savefig(outpath + title + '.png', bbox_inches = "tight", dpi = 600)
    

def sub_select(data, n, end = 'lower'): # n is the threshold for selection
    selected = pd.DataFrame(columns = columns)
    if end == 'lower':
        for i in range(n):
            gene = data.idxmin(axis = 0)[1]
            print(gene)
            selected.loc[gene] = data.min(axis = 0)
            data.drop(data.idxmin(axis = 0), axis = 0, inplace = True)
    else:
        for i in range(n):
            gene = data.idxmax(axis = 0)[1]
            print(gene)
            selected.loc[gene] = data.max(axis = 0)
            data.drop(data.idxmax(axis = 0), axis = 0, inplace = True)
    return selected;


def plot_proportions(df):
    stack_hues = ['salmon', 'powderblue', 'gold', 'plum', 'navajowhite', 'lightgrey', 'ivory', 'rosybrown']
    fig, ax = plt.subplots(figsize = (8, 6))
    lowest = [0,0,0,0,0,0]
    for r in range(8):
        plt.bar(df.columns, df.iloc[r, :], bottom = lowest, color = stack_hues[r], edgecolor = 'black', 
                    width = 0.6, linewidth = 2,label = df.index[r])
        lowest = lowest + df.iloc[r, :]
        print(df.index[r])
        
    plt.xticks(fontfamily = 'Arial', fontsize = 14, fontweight = 'bold')
    plt.yticks(fontfamily = 'Arial', fontsize = 14, fontweight = 'bold')
    ax.xaxis.set_tick_params(width = 2, length = 8)
    ax.yaxis.set_tick_params(width = 2, length = 8)
    ax.set_ylabel('Number of peaks', fontsize = 14, fontweight = 'bold')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_linewidth(2)
    ax.spines['left'].set_linewidth(2)

    handles, labels = ax.get_legend_handles_labels()
    ax.legend(reversed(handles), reversed(labels), loc = 'upper left', bbox_to_anchor = (1.05, 1), 
                  bbox_transform = ax.transAxes, frameon = False, fontsize = 14)
    ax.set_title('H3K27me3 peaks in DNA regions', fontsize = 14, fontweight = 'bold')
    plt.show()
    fig.savefig(outpath + 'DNA_region_PeakNums' + '.png', bbox_inches = "tight", dpi = 600)


def plot_PCA(df):
    df = df[columns]
    flipped = df.T
    split_sample = pd.Series(flipped.index)
    split_sample = split_sample.str.split(' r', n = 1, expand = True)
    shortened = split_sample.iloc[:,0].values
    flipped['mouse'] = shortened
    flipped.reset_index(inplace = True)
    print(flipped)
    features = []
    features.extend(range(0, 159))
    # separate features
    x = flipped.loc[:, features].values
    # standardize features
    x = StandardScaler().fit_transform(x)
    
    pca = PCA(n_components = 2)
    principalComponents = pca.fit_transform(x)
    print(pca.explained_variance_ratio_.cumsum())
    a, b = pca.explained_variance_ratio_.cumsum() * 100
    pc1 = 'PC1 (' + str(round(a, 2)) + '%)'
    pc2 = 'PC2 (' + str(round(b, 2)) + '%)'
    principalDf = pd.DataFrame(data = principalComponents, columns = [pc1, pc2])
    final = pd.concat([principalDf, flipped[['mouse']]], axis = 1)
    print(final)
    
    sns.set(font_scale = 1.2, style = 'ticks')
    fig, ax = plt.subplots(figsize = (5, 5))
    ax = sns.scatterplot(x = pc1, y = pc2, size = 'mouse', sizes = (100, 100),
                         hue = 'mouse', alpha = 1, palette = colours, data = final)
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles = handles[1:], labels = labels[1:], edgecolor = 'black', fontsize = 12, fancybox = False)
    plt.show()
    
    fig.savefig(outpath + 'PCA_Metab.png', bbox_inches = "tight", dpi = 600)
    
    
def KEGG_select(df):
    reduced = df[df['Gene'].str.contains('\(RefSeq\) ')]
    split = reduced['Gene'].str.split('\(RefSeq\) ', expand = True)
    reduced['Split'] = split.iloc[:,1]
    one = reduced['Split'].str.split(',|\\;', expand = True)
    reduced['Short'] = one.iloc[:,0]
    clean = reduced[reduced['Short'].str.len() > 2]
    one_name = clean['Short']
    one_name.drop_duplicates(inplace = True)
    one_name.reset_index(drop = True, inplace = True)
    one_name.to_csv('/Users/orsi/Desktop/Test.txt', index = False, sep = '\t')
    print(one_name)
    return one_name;


if __name__ == '__main__':
    
    # ________________________________________________________________________
    
    # this chunk for violin plot from peak files:
    
    df_pf = read_PFs('/Users/orsi/Desktop/Petrof_PeakFiles/', 7)
    df_pf = rename_cols(df_pf)
    df_pf = np.log10(df_pf)
    violin_plot(df_pf)
    
    # ________________________________________________________________________
    
    # this chunk for count and proportion of peaks in different DNA regions
    
    df = read_proportions('/Users/orsi/Desktop/Annotated/')
    df = rename_cols(df)
    print(df)
    frame = df[columns]
    frame.columns = ['WT\nr1', 'WT\nr2', 'mdx\nr1', 'mdx\nr2', 'mdx\nTLR4-/-\nr1', 'mdx\nTLR4-/-\nr2']
    plot_proportions(frame)
    
    # ________________________________________________________________________
    
    # this chunk for heatmap of tag counts
    
    df = pd.read_csv('/Users/orsi/Desktop/merged_annotatedPeaks.txt', sep = '\t')
    df = df.iloc[:,19:]
    df = rename_cols(df)
    print(df.iloc[:6,:])
    top = sub_select(df, 1000, end = 'upper')
    print(top.iloc[:10,:])
    top = top.astype(float)
    print(top.iloc[:10,:])
    top = np.log10(top + 1)
    print(top.iloc[:10,:])
    top.columns = ['WT\nr1', 'WT\nr2', 'mdx\nr1', 'mdx\nr2', 'mdx TLR4-/-\nr1', 'mdx TLR4-/-\nr2']
    heatmap(top)
    
    # ________________________________________________________________________
    
    # this chunk to select top genes for GO analysis (generic)
    
    folder = '/Users/orsi/Desktop/GO_Matrix/'
    for filename in os.listdir(folder):
        temp = pd.read_csv(folder + filename, sep = '\t')
        genes = pd.Series(temp.iloc[:,15])
        scores = pd.Series(temp.iloc[:,5])
        merge = pd.concat([genes, scores], axis = 1)
        top_200 = merge.iloc[:200,:]
        top_200.to_csv('/Users/orsi/Desktop/GO/TopGenes_' + filename, index = False, sep = '\t')
    
    # ________________________________________________________________________
    
    # this chunk for metabolism and inflammation pathway genes
    
    df1 = pd.read_csv('/Users/orsi/Desktop/Petrof Figs/GO/Petrof_InflammatoryGenes.csv')
    inflamm = KEGG_select(df1)
#    df2 = pd.read_csv('/Users/orsi/Desktop/Petrof Figs/GO/Petrof_MetabolismGenes.csv')
#    metab = KEGG_select(df2)
    
    df = pd.read_csv('/Users/orsi/Desktop/Petrof Figs/merged_annotatedPeaks.txt', sep = '\t')
    gene_col = df.iloc[:,15]
    df = df.iloc[:,19:]
    df = rename_cols(df)
    df = df[columns]
    annot = pd.concat([df, gene_col], axis = 1)
    print(annot.iloc[:6,:])
    
    gene_bool = []
    gene_names = []
    for i in range(len(df.index)):
        gene_name = gene_col[i]
        gene_names.append(gene_name)
        check = False
        for gene in inflamm:
            if gene == gene_name:
                gene_bool.append(True)
                print(gene + ' is here')
                check = True
                break
        if check == False:
            gene_bool.append(False)

    ext_df = pd.concat([df, pd.Series(gene_bool, name = 'Select'), pd.Series(gene_names, name = 'Gene')], axis = 1)
    clean_df = ext_df[ext_df['Select'] == True]
    clean_genes = clean_df['Gene']
    clean_df.drop(['Select'], axis = 1, inplace = True)
    grouped = clean_df.groupby('Gene').mean()
    grouped.index.name = None
    grouped.to_csv('/Users/orsi/Desktop/OxphosGenes_Raw.txt', index = True, sep = '\t')
    
    read_df = pd.read_csv('/Users/orsi/Desktop/OxphosGenes_Raw.txt', sep = '\t')
    print(read_df)
    df = read_df.loc[:, ~read_df.columns.str.contains('^Unnamed')]
    print(df)
    df.columns = ['WT\nr1', 'WT\nr2', 'mdx\nr1', 'mdx\nr2', 'mdx\nTLR4-/-\nr1', 'mdx\nTLR4-/-\nr2']
    df = df[df.min(axis = 1) > 10] # remove rows with weak signals
    df.reset_index(inplace = True, drop = True)
    df = np.log10(df)
    print(df)
    heatmap_large(df, 'Heatmap')
    
    swarm_plot(df, 'Swarm')
    
    plot_PCA(df)
    
    # testing significance between groups
    
    import itertools
    c = list(itertools.combinations(range(len(df.columns)), r = 2))
    output_df = pd.DataFrame(columns = ['Comparison', 'Stat', 'Pval', 'Sig'])
    for pair in c:
        stat, pval = sci.mannwhitneyu(df.iloc[:,pair[0]], df.iloc[:,pair[1]])
        sig = ''
        if pval <= 0.001:
            sig = '***'
        elif pval <= 0.01:
            sig = '**'
        elif pval <= 0.05:
            sig = '*'
        comp = df.columns[pair[0]] + ' vs. ' + df.columns[pair[1]]
        new_row = [comp, stat, pval, sig]
        output_df.loc[len(output_df)] = new_row 
    print(output_df)
    output_df.to_csv('/Users/orsi/Desktop/OxphosGenes_Sig.csv', index = False)
    
    # ________________________________________________________________________
    
    # scatterplots for tag counts (rep 1 vs. rep 2)
    
    df = pd.read_csv('/Users/orsi/Desktop/merged_annotatedPeaks.txt', sep = '\t')
    df = df.iloc[:,19:]
    df = rename_cols(df)
    frame = df[columns]
    print(frame.iloc[:6,:])
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize = (12, 4))
    counter = 0
    for ax in fig.get_axes():
        xname = frame.columns[counter]
        yname = frame.columns[counter + 1]
        print(xname + yname)
        ax.scatter(frame.iloc[:, counter], frame.iloc[:, counter + 1], color = hues[counter], s = 1)
        ax.set_xlabel(xname, fontsize = 14)
        ax.set_ylabel(yname, fontsize = 14)
        counter += 2
    fig.suptitle('Tag counts in H3K27me3 peaks between replicates', fontsize = 14)
    fig.tight_layout(rect=[0, 0.03, 1, 0.90])
    fig.savefig(outpath + 'Scatter_Reps' + '.png', bbox_inches = "tight", dpi = 600)
    plt.show()
    
    # ________________________________________________________________________
    

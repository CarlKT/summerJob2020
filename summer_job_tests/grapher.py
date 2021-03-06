import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import itertools


#Instantiate usable variables
class Grapher:
    """takes a dataframe"""
    def __init__(self, _input, figsize):
        self._input = _input
        self.dtype = type(_input)
        self.fig = plt.figure(figsize=figsize)

        #make sure the input is a DataFrame
        if self.dtype != type(pd.DataFrame()):
            raise TypeError(
                'grapher input must be a DataFrame, got %s instead' %
                self.dtype)

    def heatmap(self):
        _input, fig = self._input, self.fig
        ax = fig.subplots()
        ax = sns.heatmap(_input)

    def violin_plot(self):
        _input, fig = self._input, self.fig
        ax = fig.subplots()
        ax = sns.violinplot(_input)

    def scatter(self, cols):
        _input, fig = self._input, self.fig
        ax = fig.subplots()
        ax = sns.scatterplot(_input.iloc[:, cols[0]], _input.iloc[:, cols[1]])

    def big_scatter(self,
                    cols,
                    shape,
                    fixed_x=None,
                    fixed_y=None,
                    axis_names=[],
                    fig_name=''):
        """makes a series of scatter plots using any number of variables.
        By default, it graphs every combination of variables, but it can also
        support a fixed x axis or y axis."""

        _input, fig = self._input, self.fig
        axes = fig.subplots(nrows=shape[0], ncols=shape[1], squeeze=True)
        axes = axes.flatten()
        #instantiate colors
        colors = [
            '#f54242', '#ef42f5', '#56dbce', '#ffa600', '#9c00fc', '#00fc2a',
            '#eaf202'
        ]

        #indentify columns to use and turn into a dictionnary
        features = []
        if len(axis_names) == len(cols) or len(axis_names) == 0:
            feature_names = axis_names
            for col in cols:
                features.append(_input.iloc[:, col])
                feature_names.append(_input.columns[col])
        else:
            raise ValueError(
                'axis_names not the correct length (must be 0 or the same length as cols)'
            )

        feature_dict = {
            feature_names[i]: features[i]
            for i in range(len(cols))
        }

        #Generate combinations
        feat_combi = list(itertools.combinations(feature_names, 2))

        #loop through axes and plot the proper cols
        index = 0
        for ax in axes:

            #Use different variables based on what is fixed
            if fixed_x == None and fixed_y == None:
                X_name = feat_combi[index][0]
                y_name = feat_combi[index][1]
            elif fixed_x != None:
                X_name = feature_names[fixed_x]
                y_name = feature_names[index + 1]
            elif fixed_y != None:
                X_name = feature_names[index + 1]
                y_name = feature_names[fixed_y]
            else:
                raise ValueError('Use scatter for 2 variable plotting')

            X, y = feature_dict[X_name], feature_dict[y_name]

            ax.scatter(X, y, c=colors[index], s=0.5)

            ax.set_xlabel(X_name)
            ax.set_ylabel(y_name)
            index += 1
        fig.suptitle(fig_name)


#Just for testing stuff
if __name__ == '__main__':

    out_dir = '/Users/carlkristoftessier/Documents/all_that_programming/summerJob2020/Figures/big_scatter.png'
    df = pd.DataFrame([[1, 0, 0], [1, 1, 0], [0, 0, 1]])
    grapher = Grapher(df, (12, 3))
    grapher.big_scatter((0, 1, 2), (2, 2), fig_name='test')
    plt.show()
    #grapher.fig.savefig(out_dir, bbox_inches="tight", dpi=600)

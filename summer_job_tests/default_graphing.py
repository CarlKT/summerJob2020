import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from grapher import Grapher

#Import data
txt_dir = '/Users/carlkristoftessier/Documents/all_that_programming/summerJob2020/merged_annotatedPeaks.txt'
df = pd.read_csv(txt_dir, sep='\t')

#Preprocessing
peak_score = pd.DataFrame(df['Peak Score'])
tag_data = pd.DataFrame([df.iloc[:, -i] for i in range(1, 7)]).T
data_to_plot = pd.concat([peak_score, tag_data], axis=1)
log_data = np.log10(data_to_plot)

#Instantiating graph variables
figsize = (12, 6)
fig_name = 'Peak score for each motif tag count with log scaling'
axis_names = [
    'Peak Score', 'CCGTCC tag count (in bp)', 'ATGTCA tag count (in bp)',
    'AGTTCC tag count (in bp)', 'AGTCAA tag count (in bp)',
    'GTCCGC tag count (in bp)', 'GTAGAG tag count (in bp)'
]
cols_to_graph = list(range(7))

#Create Grapher object and make a big scatter
grapher = Grapher(log_data, figsize)
grapher.big_scatter(cols_to_graph,
                    shape=(2, 3),
                    fixed_y=0,
                    axis_names=axis_names,
                    fig_name=fig_name)

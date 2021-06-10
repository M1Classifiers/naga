import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from pandas.api.types import is_string_dtype, is_numeric_dtype

def univariate_visualization(df):
    sns.set(style="ticks")
    for col in df:
        if is_numeric_dtype(df[col]):
            f, (ax_box, ax_hist) = plt.subplots(2, sharex=True, gridspec_kw={"height_ratios": (.10, .90)}, figsize = (12, 12))
            sns.boxplot(df[col], ax=ax_box)
            sns.distplot(df[col], ax=ax_hist, kde =False)
            ax_box.set(yticks=[])
            sns.despine(ax=ax_hist)
            sns.despine(ax=ax_box, left=True)
            ax_hist.set_ylabel('Frequency')
            ax_box.set_xlabel(None)
        elif is_string_dtype(df[col]):
            if len(df[col].value_counts()) > 1:
                max_char_len = 0

                plt.figure(figsize = (len(df[col].unique()) * 0.75, 7))
                df[col].value_counts().plot(kind = 'bar', fontsize = 25)
                plt.ylabel('Frequency', fontsize = 25)          
    plt.show()
    
    def plot(csv_loc):
    df = pd.read_csv(csv_loc, index_col = 0)
    univariate_visualization(df)

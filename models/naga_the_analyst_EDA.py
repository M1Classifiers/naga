import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from pandas.api.types import is_string_dtype, is_numeric_dtype, is_bool_dtype

def num_dtype_checker(df):  
    ''' converts should-be-numeric dtype from string dtype; recognized by pd.read_csv as string due to comma'''
    for col in df:
        if is_string_dtype(df[col]):
            if df[df[col].notnull()][col].apply(lambda x: len(x.split(',')) > 1).any():
                col_values = np.zeros(len(df[col]))
                for ind, num in enumerate(df[col]):
                    new_val = ''
                    if len(num.split(',')) > 1:
                        for char in num.split(','):
                            new_val += char
                        col_values[ind] = new_val
                    elif type(num) == str:
                        col_values[ind] = num
                    else:
                        col_values[ind] = np.nan
                df[col] = col_values
            else:
                continue

def index_col_duplicate_checker(df, col):
    for prev, current in zip(df[col][:-1], df[col][1:]):
        if current - prev == 1 :
            continue
        else:
            return False
    return True

def drop_duplicate_index_col(df):
    for col in df:
        if is_string_dtype(df[col]) or is_bool_dtype(df[col]):
            continue
        if index_col_duplicate_checker(df, col):
            df.drop(columns = col, inplace = True)

def univariate_visualization(df, save_to, save_as):
    sns.set(style="ticks")
    for col in df:
        if is_numeric_dtype(df[col]):
            f, (ax_box, ax_hist) = plt.subplots(2, sharex=True, gridspec_kw={"height_ratios": (.10, .90)})
            sns.boxplot(df[col], ax=ax_box)
            sns.distplot(df[col], ax=ax_hist, kde =False)
            ax_box.set(yticks=[])
            sns.despine(ax=ax_hist)
            sns.despine(ax=ax_box, left=True)
            ax_hist.set_ylabel('Frequency')
            ax_box.set_xlabel(None)
            plt.savefig(save_to + save_as + '.jpeg')

        elif is_string_dtype(df[col]):
            if len(df[col].value_counts()) > 1:
                ax = sns.countplot(x = df_raw.Region)
                for p in ax.patches:
                    ax.annotate('{}'.format(p.get_height()), (p.get_x()+0.2, p.get_height()/2))
                plt.savefig(save_to + save_as + '.jpeg')
    #plt.show()
    
def plot(csv_loc, save_to, save_as):
    df = pd.read_csv(csv_loc)
    num_dtype_checker(df)
    drop_duplicate_index_col(df)
    univariate_visualization(df, save_to, save_as)
    

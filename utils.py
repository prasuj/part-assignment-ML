import pandas as pd
import numpy as np


def entropy(Y: pd.Series) -> float:
    entrop = 0
    fraction = Y.value_counts()/len(Y)
    entrop = - fraction.apply(lambda x: x * np.log2(x)).sum()
    return entrop
  
def gini_index(Y: pd.Series) -> float:
    """
    Function to calculate the gini index
    """
    pass

def information_gain(Y: pd.Series, attr: pd.Series) -> float:
    total_samples = Y.shape[0]
    df_out = pd.crosstab(attr,Y)
    values_in_attributes = df_out.index
    frequency_of_values = df_out.sum(axis=1)
    wt = frequency_of_values.apply(lambda x: x / total_samples)
    l = []
    df = pd.DataFrame({
        "attr": attr,
        "y": Y
    })
    for v in attr.unique():
        l.append(entropy(df[df["attr"] == v]["y"]))
    l = pd.Series(l)
    return wt.to_numpy().dot(l.to_numpy())

    




    
    

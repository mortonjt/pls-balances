import pandas as pd
import numpy as np
from biom import Table, load_table

metadata = pd.read_table("pls_balances/data/globalpatterns/GlobalPatterns_mapping.txt", index_col=0)

table = load_table("pls_balances/data/globalpatterns/globalpatterns_s1_x13_filter.biom")

table = pd.DataFrame(np.array(table.matrix_data.todense()).T,
                     index=table.ids(axis='sample'),
                     columns=table.ids(axis='observation'))

sample = table.loc[['CL3'],:]
x = sample.loc['CL3',:] > 0

sample_no_zeroes = sample[list(x[x].index)]

sample_no_zeroes = np.array(sample_no_zeroes.values)[0]


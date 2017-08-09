from biom import load_table
import pandas as pd


def compute_confusion_matrices(table_files, result_files, truth_files, output_file):
    stats = pd.DataFrame(index=table_files, columns=['TP', 'FP', 'FN', 'TN'])
    for tab_file, r_file, t_file in zip(table_files,
                                        result_files,
                                        truth_files):

        hits = set(open(r_file, 'r').read().split(','))
        truth = set(open(t_file, 'r').read().split(','))
        ids = set(load_table(tab_file).ids(axis='observation'))

        x = pd.Series(
            {'TP': len(hits & truth),
             'FP': len(hits - truth),
             'FN': len(truth - hits),
             'TN': len((ids-hits) & (ids-truth))
            })
        stats.loc[tab_file] = x
    stats.to_csv(output_file, sep='\t')

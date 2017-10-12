from biom import load_table
import pandas as pd
from os.path import basename, splitext


def compute_confusion_matrices(table_files, result_files,
                               truth_files, output_file):
    """ Computes confusion matrice over all runs for a
    specified set of results.

    Parameters
    ----------
    table_files : list of str
        List of filepaths for biom tables.
    result_files : list of str
        List of filepaths for estimated differentially abundant features.
    truth_files : list of str
        List of filepaths for ground truth differentially abundant features.
    output_file : str
        File path for confusion matrix summary.

    Note
    ----
    This assumes that the tables all have the same basename.
    """
    # only use the result files that match with the output_file
    out_suf = splitext(basename(output_file))[0]

    result_files = list(filter(lambda x: out_suf in basename(x), result_files))
    index_names = list(map(lambda x: splitext(basename(x))[0], table_files))
    col_names = ['%s_TP' % out_suf,
                 '%s_FP' % out_suf,
                 '%s_FN' % out_suf,
                 '%s_TN' % out_suf]
    TP, FP, FN, TN = 0, 1, 2, 3

    stats = pd.DataFrame(columns=col_names, index=table_files)
    for tab_file, r_file, t_file in zip(table_files,
                                        result_files,
                                        truth_files):

        hits = set(open(r_file, 'r').read().split(','))
        truth = set(open(t_file, 'r').read().split(','))
        ids = set(load_table(tab_file).ids(axis='observation'))

        x = pd.Series(
            {col_names[TP]: len(hits & truth),
             col_names[FP]: len(hits - truth),
             col_names[FN]: len(truth - hits),
             col_names[TN]: len((ids-hits) & (ids-truth))
            })

        stats.loc[tab_file, :] = x

    stats.to_csv(output_file, sep='\t')


def aggregate_summaries(confusion_matrix_files, table_files, metadata_files,
                        axis, output_file):
    """ Aggregates summary files together, along with the variable of interest.

    Parameters
    ----------
    confusion_matrix_files : list of str
        List of filepaths for summaries.
    table_files : list of str
        List of filepaths for biom tables.
    metadata_files : list of str
        List of filepaths for metadata files.
    axis : str
        Category of differentiation.
    output_file : str
        Output path for aggregated summaries.

    Note
    ----
    This assumes that table_files and metadata_files
    are in the same matching order.

    table.xxx.biom
    metadata.xxx.txt
    """
    mats = [pd.read_table(f, index_col=0) for f in confusion_matrix_files]
    merged_stats = pd.concat(mats, axis=1)

    # first
    index_names = table_files
    # aggregate stats in all metadata_files. For now, just take the mean
    x = [pd.read_table(f, index_col=0)[axis].mean() for f in metadata_files]
    cats = pd.DataFrame(x, index=index_names, columns=[axis])
    merged_stats = pd.merge(merged_stats, cats, left_index=True, right_index=True)
    merged_stats.to_csv(output_file, sep='\t')

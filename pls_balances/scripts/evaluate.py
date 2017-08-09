import click


@click.group()
def evaluate():
    pass


@evaluate.command()
@click.argument('--table-files', nargs=-1)
@click.argument('--result-files', nargs=-1)
@click.argument('--truth-files', nargs=-1)
@click.option('--output-file')
def compute_confusion_matrices(table_files, result_files, truth_files, output_file):
    stats = pd.DataFrame(columns=['TP', 'FP', 'FN', 'TN'])

    for tab_file, r_file, t_file in zip(table_files,
                                        result_files,
                                        truth_files):

        hits = set(open(r_file, 'r').read().split(','))
        truth = set(open(t_file, 'r').read().split(','))
        ids = set(load_table(tab_file).ids(axis='observation'))

        stats.append(
            pd.Series(
                {'TP': hits & truth,
                 'FP': hits - truth,
                 'FN': truth - hits,
                 'TN': (ids-hits) & (ids-truth)
                },
                name=tab_file)
        )
    stats.to_csv(output_file, sep='\t')


if __name__ == "__main__":
    evaluate()

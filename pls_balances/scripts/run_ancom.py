import pandas as pd
from skbio.stats.composition import ancom
import click
from biom import load_table
@click.command()
@click.option('--table-file')
@click.option('--category-file')
@click.option('--output-file')
def run_ancom(table_file, category_file, output_file):
    category = pd.Series.from_csv(category_file)
    table = load_table(table)
    table = pd.DataFrame(np.array(table.matrix_data.todense()).T,
                         index=table.ids(axis='sample'),
                         columns=table.ids(axis='observation'))
    res, _ = ancom(table, metadata[category])
    with open(output, 'w') as f:
        r = res["Reject null hypothesis"]
        f.write(','.join(res.loc[r].index.values))

if __name__ == '__main__':
    run_ancom()

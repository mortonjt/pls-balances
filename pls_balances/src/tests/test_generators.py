import unittest
import numpy as np
import pandas as pd
import pandas.util.testing as pdt
from skbio.stats.composition import closure
from pls_balances.src.generators import (compositional_effect_size_generator,
                                         compositional_variable_features_generator)


class TestCompositionalEffectSize(unittest.TestCase):

    def test_composition_effect_size_simple(self):

        gen = compositional_effect_size_generator(max_alpha=1, reps=5,
                                                  intervals=2, n_species=5)
        table, metadata, truth = next(gen)

        exp_table = pd.DataFrame(
            np.vstack((
                np.array([1, 1, 1, 1, 1]) / 5,
                np.array([1, 1, 1, 1, 1]) / 5,
                np.array([1, 1, 1, 1, 1]) / 5,
                np.array([1, 1, 1, 1, 1]) / 5,
                np.array([1, 1, 1, 1, 1]) / 5,
                np.array([1, 1, 1, 1, 1]) / 5,
                np.array([1, 1, 1, 1, 1]) / 5,
                np.array([1, 1, 1, 1, 1]) / 5,
                np.array([1, 1, 1, 1, 1]) / 5,
                np.array([1, 1, 1, 1, 1]) / 5
            )),
            index = ['S0', 'S1', 'S2', 'S3', 'S4',
                     'S5', 'S6', 'S7', 'S8', 'S9'],
            columns = ['F0', 'F1', 'F2', 'F3', 'F4']
        )

        pdt.assert_frame_equal(table, exp_table)
        exp_metadata = pd.DataFrame(
            {'group': [0] * 5 + [1] * 5,
             'n_diff': [2] * 10,
             'effect_size': [1.0] * 10,
             'library_size': [10000] * 10
            },
            index = ['S0', 'S1', 'S2', 'S3', 'S4',
                     'S5', 'S6', 'S7', 'S8', 'S9'],
        )
        metadata = metadata.reindex_axis(sorted(metadata.columns), axis=1)
        exp_metadata = exp_metadata.reindex_axis(sorted(exp_metadata.columns), axis=1)
        pdt.assert_frame_equal(metadata, exp_metadata)

        exp_truth = ['F0', 'F4']
        self.assertListEqual(truth, exp_truth)

        # test to see if the groups are different
        table, metadata, truth = next(gen)

        exp_table = pd.DataFrame(
            closure(
                np.vstack((
                    np.array([10, 1, 1, 1, 1]),
                    np.array([10, 1, 1, 1, 1]),
                    np.array([10, 1, 1, 1, 1]),
                    np.array([10, 1, 1, 1, 1]),
                    np.array([10, 1, 1, 1, 1]),
                    np.array([1, 1, 1, 1, 10]),
                    np.array([1, 1, 1, 1, 10]),
                    np.array([1, 1, 1, 1, 10]),
                    np.array([1, 1, 1, 1, 10]),
                    np.array([1, 1, 1, 1, 10])
                ))),
            index = ['S0', 'S1', 'S2', 'S3', 'S4',
                     'S5', 'S6', 'S7', 'S8', 'S9'],
            columns = ['F0', 'F1', 'F2', 'F3', 'F4']
        )

        pdt.assert_frame_equal(table, exp_table)

        exp_metadata = pd.DataFrame(
            {'group': [0] * 5 + [1] * 5,
             'n_diff': [2] * 10,
             'effect_size': [10.0] * 10,
             'library_size': [10000] * 10
            },
            index = ['S0', 'S1', 'S2', 'S3', 'S4',
                     'S5', 'S6', 'S7', 'S8', 'S9'],
        )
        metadata = metadata.reindex_axis(sorted(metadata.columns), axis=1)
        exp_metadata = exp_metadata.reindex_axis(sorted(exp_metadata.columns), axis=1)

        pdt.assert_frame_equal(metadata, exp_metadata)

        exp_truth = ['F0', 'F4']
        self.assertListEqual(truth, exp_truth)

    def test_composition_variable_features(self):
        gen = compositional_variable_features_generator(
            max_changing=2, fold_change=2, reps=5,
            intervals=2, n_species=5)

        table, metadata, truth = next(gen)
        table, metadata, truth = next(gen)
        exp_table = pd.DataFrame(
            closure(
                np.vstack((
                    np.array([2, 2, 1, 1, 1]),
                    np.array([2, 2, 1, 1, 1]),
                    np.array([2, 2, 1, 1, 1]),
                    np.array([2, 2, 1, 1, 1]),
                    np.array([2, 2, 1, 1, 1]),
                    np.array([1, 1, 1, 2, 2]),
                    np.array([1, 1, 1, 2, 2]),
                    np.array([1, 1, 1, 2, 2]),
                    np.array([1, 1, 1, 2, 2]),
                    np.array([1, 1, 1, 2, 2])
                ))
            ),
            index = ['S0', 'S1', 'S2', 'S3', 'S4',
                     'S5', 'S6', 'S7', 'S8', 'S9'],
            columns = ['F0', 'F1', 'F2', 'F3', 'F4']
        )
        pdt.assert_frame_equal(table, exp_table)

        exp_metadata = pd.DataFrame(
            {'group': [0] * 5 + [1] * 5,
             'n_diff': [4] * 10,
             'effect_size': [2] * 10,
             'library_size': [10000] * 10
            },
            index = ['S0', 'S1', 'S2', 'S3', 'S4',
                     'S5', 'S6', 'S7', 'S8', 'S9'],
        )

        metadata = metadata.reindex_axis(sorted(metadata.columns), axis=1)
        exp_metadata = exp_metadata.reindex_axis(sorted(exp_metadata.columns), axis=1)
        pdt.assert_frame_equal(metadata, exp_metadata)

        exp_truth = pd.Series(
            [1, 1, 0, 1, 1]
        )

        exp_truth = ['F0', 'F1', 'F3', 'F4']
        self.assertListEqual(truth, exp_truth)


if __name__ == "__main__":
    unittest.main()

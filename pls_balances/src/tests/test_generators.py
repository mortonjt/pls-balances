import unittest
import numpy as np
import pandas as pd
import pandas.util.testing as pdt
import numpy.testing as npt
from skbio.stats.composition import closure
from pls_balances.src.generators import (compositional_effect_size_generator,
                                         compositional_variable_features_generator,
                                         compositional_regression_prefilter_generator)


class TestCompositionalEffectSize(unittest.TestCase):

    def test_composition_effect_size_simple(self):

        gen = compositional_effect_size_generator(max_alpha=1, reps=5,
                                                  intervals=2, n_species=5, n_diff=1)
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


class TestCompositionalRegression(unittest.TestCase):
    def test_compositional_regression_prefilter_generator(self):

        gen = compositional_regression_prefilter_generator(
            max_gradient=5, gradient_intervals=10, sigma=1,
            n_species=5, lam=0.1, max_contaminants=3,
            contaminant_intervals=2
        )
        table, md, truth = next(gen)

        exp_table = pd.DataFrame({
            'S0': [1.663719e-01, 0.076171, 0.007310, 0.000147, 6.200103e-07,
                   0.249989, 0.000011, 0.499977, 0.000023],
            'S1': [1.190353e-01, 0.109138, 0.020975, 0.000845, 7.134577e-06,
                   0.249989, 0.000011, 0.499977, 0.000023],
            'S2': [6.943075e-02, 0.127481, 0.049063, 0.003958, 6.692956e-05,
                   0.249989, 0.000011, 0.499977, 0.000023],
            'S3': [3.131175e-02, 0.115132, 0.088736, 0.014336, 4.854528e-04,
                   0.249989, 0.000011, 0.499977, 0.000023],
            'S4': [1.056862e-02, 0.077821, 0.120114, 0.038860, 2.635308e-03,
                   0.249989, 0.000011, 0.499977, 0.000023],
            'S5': [2.635308e-03, 0.038860, 0.120114, 0.077821, 1.056862e-02,
                   0.249989, 0.000011, 0.499977, 0.000023],
            'S6': [4.854528e-04, 0.014336, 0.088736, 0.115132, 3.131175e-02,
                   0.249989, 0.000011, 0.499977, 0.000023],
            'S7': [6.692956e-05, 0.003958, 0.049063, 0.127481, 6.943075e-02,
                   0.249989, 0.000011, 0.499977, 0.000023],
            'S8': [7.134577e-06, 0.000845, 0.020975, 0.109138, 1.190353e-01,
                   0.249989, 0.000011, 0.499977, 0.000023],
            'S9': [6.200103e-07, 0.000147, 0.007310, 0.076171, 1.663719e-01,
                   0.249989, 0.000011, 0.499977, 0.000023]},
            index=['F0', 'F1', 'F2', 'F3', 'F4', 'F5', 'F6', 'F7', 'F8']).T

        npt.assert_allclose(table.values, exp_table.values, atol=1e-3)

        exp_md = pd.DataFrame(
            {'S0': [0.000000, 5, 10000],
             'S1': [0.555556, 5, 10000],
             'S2': [1.111111, 5, 10000],
             'S3': [1.666667, 5, 10000],
             'S4': [2.222222, 5, 10000],
             'S5': [2.777778, 5, 10000],
             'S6': [3.333333, 5, 10000],
             'S7': [3.888889, 5, 10000],
             'S8': [4.444444, 5, 10000],
             'S9': [5.000000, 5, 10000]},
            index=['gradient', 'n_diff', 'library_size']).T

        md = md.astype(np.float)

        pdt.assert_frame_equal(md, exp_md)

        exp_truth = ['F0', 'F1', 'F2', 'F3', 'F4']

        self.assertListEqual(truth, exp_truth)


if __name__ == "__main__":
    unittest.main()

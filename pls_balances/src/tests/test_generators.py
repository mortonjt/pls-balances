import unittest
import numpy as np
import pandas as pd
import pandas.util.testing as pdt
import numpy.testing as npt
from skbio.stats.composition import closure
from pls_balances.src.generators import (
    compositional_effect_size_generator,
    compositional_variable_features_generator,
    compositional_regression_prefilter_generator,
    compositional_regression_effect_size_generator)


class TestCompositionalEffectSize(unittest.TestCase):

    def test_composition_effect_size_simple(self):

        gen = compositional_effect_size_generator(max_alpha=1, reps=5,
                                                  intervals=2, n_species=5, n_diff=1,
                                                  n_contaminants=2, lam=0.1)
        table, metadata, truth = next(gen)

        exp_table = pd.DataFrame(
            np.vstack((
                np.array([.1, .1, .1, .1, .1, 0.499977, 0.00002269]),
                np.array([.1, .1, .1, .1, .1, 0.499977, 0.00002269]),
                np.array([.1, .1, .1, .1, .1, 0.499977, 0.00002269]),
                np.array([.1, .1, .1, .1, .1, 0.499977, 0.00002269]),
                np.array([.1, .1, .1, .1, .1, 0.499977, 0.00002269]),
                np.array([.1, .1, .1, .1, .1, 0.499977, 0.00002269]),
                np.array([.1, .1, .1, .1, .1, 0.499977, 0.00002269]),
                np.array([.1, .1, .1, .1, .1, 0.499977, 0.00002269]),
                np.array([.1, .1, .1, .1, .1, 0.499977, 0.00002269]),
                np.array([.1, .1, .1, .1, .1, 0.499977, 0.00002269])
            )),
            index = ['S0', 'S1', 'S2', 'S3', 'S4',
                     'S5', 'S6', 'S7', 'S8', 'S9'],
            columns = ['F0', 'F1', 'F2', 'F3', 'F4', 'X0', 'X1']
        )
        pdt.assert_frame_equal(table, exp_table, check_less_precise=True)
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
                    np.array([0.357143] + [0.035714]*4 +
                             [0.499977, 0.00002269]),
                    np.array([0.357143] + [0.035714]*4 +
                             [0.499977, 0.00002269]),
                    np.array([0.357143] + [0.035714]*4 +
                             [0.499977, 0.00002269]),
                    np.array([0.357143] + [0.035714]*4 +
                             [0.499977, 0.00002269]),
                    np.array([0.357143] + [0.035714]*4 +
                             [0.499977, 0.00002269]),
                    np.array([0.035714]*4 + [0.357143] +
                             [0.499977, 0.00002269]),
                    np.array([0.035714]*4 + [0.357143] +
                             [0.499977, 0.00002269]),
                    np.array([0.035714]*4 + [0.357143] +
                             [0.499977, 0.00002269]),
                    np.array([0.035714]*4 + [0.357143] +
                             [0.499977, 0.00002269]),
                    np.array([0.035714]*4 + [0.357143] +
                             [0.499977, 0.00002269])
                ))),
            index = ['S0', 'S1', 'S2', 'S3', 'S4',
                     'S5', 'S6', 'S7', 'S8', 'S9'],
            columns = ['F0', 'F1', 'F2', 'F3', 'F4', 'X0', 'X1']
        )

        pdt.assert_frame_equal(table, exp_table, check_less_precise=True)

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

    def test_composition_effect_size_balanced(self):

        gen = compositional_effect_size_generator(max_alpha=1, reps=5,
                                                  intervals=2, n_species=5, n_diff=1,
                                                  n_contaminants=2, lam=0.1,
                                                  balanced=False)
        table, metadata, truth = next(gen)
        table, metadata, truth = next(gen)

        exp_table = pd.DataFrame({
            'S0': [0.100000, 0.100000, 0.100000, 0.100000, 0.100000,
                   0.499977, 0.0000226989],
            'S1': [0.100000, 0.100000, 0.100000, 0.100000, 0.100000,
                   0.499977, 0.0000226989],
            'S2': [0.100000, 0.100000, 0.100000, 0.100000, 0.100000,
                   0.499977, 0.0000226989],
            'S3': [0.100000, 0.100000, 0.100000, 0.100000, 0.100000,
                   0.499977, 0.0000226989],
            'S4': [0.100000, 0.100000, 0.100000, 0.100000, 0.100000,
                   0.499977, 0.0000226989],
            'S5': [0.003817, 0.038168, 0.038168, 0.038168, 0.381679,
                   0.499977, 0.0000226989],
            'S6': [0.003817, 0.038168, 0.038168, 0.038168, 0.381679,
                   0.499977, 0.0000226989],
            'S7': [0.003817, 0.038168, 0.038168, 0.038168, 0.381679,
                   0.499977, 0.0000226989],
            'S8': [0.003817, 0.038168, 0.038168, 0.038168, 0.381679,
                   0.499977, 0.0000226989],
            'S9': [0.003817, 0.038168, 0.038168, 0.038168, 0.381679,
                   0.499977, 0.0000226989]},
            index=['F0', 'F1', 'F2', 'F3', 'F4', 'X0', 'X1']
        ).T

        pdt.assert_frame_equal(table, exp_table, check_less_precise=True)
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
            intervals=2, n_species=5,
            n_contaminants=2, lam=0.1)

        table, metadata, truth = next(gen)
        table, metadata, truth = next(gen)

        exp_table = pd.DataFrame(
            closure(
                np.vstack((
                    np.array([0.142857]*2 + [0.071429]*3 +
                             [0.499977, 0.00002269]),
                    np.array([0.142857]*2 + [0.071429]*3 +
                             [0.499977, 0.00002269]),
                    np.array([0.142857]*2 + [0.071429]*3 +
                             [0.499977, 0.00002269]),
                    np.array([0.142857]*2 + [0.071429]*3 +
                             [0.499977, 0.00002269]),
                    np.array([0.142857]*2 + [0.071429]*3 +
                             [0.499977, 0.00002269]),
                    np.array([0.071429]*3 + [0.142857]*2 +
                             [0.499977, 0.00002269]),
                    np.array([0.071429]*3 + [0.142857]*2 +
                             [0.499977, 0.00002269]),
                    np.array([0.071429]*3 + [0.142857] *2+
                             [0.499977, 0.00002269]),
                    np.array([0.071429]*3 + [0.142857]*2 +
                             [0.499977, 0.00002269]),
                    np.array([0.071429]*3 + [0.142857]*2 +
                             [0.499977, 0.00002269])
                ))),
            index = ['S0', 'S1', 'S2', 'S3', 'S4',
                     'S5', 'S6', 'S7', 'S8', 'S9'],
            columns = ['F0', 'F1', 'F2', 'F3', 'F4', 'X0', 'X1']
        )

        pdt.assert_frame_equal(table, exp_table, check_less_precise=True)

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
        # first table
        table, md, truth = next(gen)

        exp_table = pd.DataFrame({
            'S0': [0.332744, 0.152341, 0.014620, 0.000294, 0.000001,
                   0.499977, 0.000023],
            'S1': [0.238071, 0.218276, 0.041949, 0.001690, 0.000014,
                   0.499977, 0.000023],
            'S2': [0.138861, 0.254962, 0.098126, 0.007916, 0.000134,
                   0.499977, 0.000023],
            'S3': [0.062624, 0.230263, 0.177471, 0.028671, 0.000971,
                   0.499977, 0.000023],
            'S4': [0.021137, 0.155643, 0.240229, 0.077721, 0.005271,
                   0.499977, 0.000023],
            'S5': [0.005271, 0.077721, 0.240229, 0.155643, 0.021137,
                   0.499977, 0.000023],
            'S6': [0.000971, 0.028671, 0.177471, 0.230263, 0.062624,
                   0.499977, 0.000023],
            'S7': [0.000134, 0.007916, 0.098126, 0.254962, 0.138861,
                   0.499977, 0.000023],
            'S8': [0.000014, 0.001690, 0.041949, 0.218276, 0.238071,
                   0.499977, 0.000023],
            'S9': [0.000001, 0.000294, 0.014620, 0.152341, 0.332744,
                   0.499977, 0.000023]
        }, ['F0', 'F1', 'F2', 'F3', 'F4', 'X0', 'X1']).T
        npt.assert_allclose(table.values, exp_table.values, atol=1e-3)

        exp_md = pd.DataFrame(
            {'S0': [0.000000, 5, 10000, 1.0, 2],
             'S1': [0.555556, 5, 10000, 1.0, 2],
             'S2': [1.111111, 5, 10000, 1.0, 2],
             'S3': [1.666667, 5, 10000, 1.0, 2],
             'S4': [2.222222, 5, 10000, 1.0, 2],
             'S5': [2.777778, 5, 10000, 1.0, 2],
             'S6': [3.333333, 5, 10000, 1.0, 2],
             'S7': [3.888889, 5, 10000, 1.0, 2],
             'S8': [4.444444, 5, 10000, 1.0, 2],
             'S9': [5.000000, 5, 10000, 1.0, 2]},
            index=['gradient', 'n_diff', 'library_size',
                   'effect_size', 'n_contaminants']).T

        md = md.astype(np.float)
        md = md.reindex(columns=['gradient', 'n_diff', 'library_size',
                                 'effect_size', 'n_contaminants'])
        pdt.assert_frame_equal(md, exp_md)

        exp_truth = ['F0', 'F1', 'F2', 'F3', 'F4']

        self.assertListEqual(truth, exp_truth)

        # second table
        table, md, truth = next(gen)

        exp_table = pd.DataFrame({
            'S0': [0.332744, 0.152341, 0.014620, 0.000294, 0.000001,
                   0.496631, 0.003346, 0.000023],
            'S1': [0.238071, 0.218276, 0.041949, 0.001690, 0.000014,
                   0.496631, 0.003346, 0.000023],
            'S2': [0.138861, 0.254962, 0.098126, 0.007916, 0.000134,
                   0.496631, 0.003346, 0.000023],
            'S3': [0.062624, 0.230263, 0.177471, 0.028671, 0.000971,
                   0.496631, 0.003346, 0.000023],
            'S4': [0.021137, 0.155643, 0.240229, 0.077721, 0.005271,
                   0.496631, 0.003346, 0.000023],
            'S5': [0.005271, 0.077721, 0.240229, 0.155643, 0.021137,
                   0.496631, 0.003346, 0.000023],
            'S6': [0.000971, 0.028671, 0.177471, 0.230263, 0.062624,
                   0.496631, 0.003346, 0.000023],
            'S7': [0.000134, 0.007916, 0.098126, 0.254962, 0.138861,
                   0.496631, 0.003346, 0.000023],
            'S8': [0.000014, 0.001690, 0.041949, 0.218276, 0.238071,
                   0.496631, 0.003346, 0.000023],
            'S9': [0.000001, 0.000294, 0.014620, 0.152341, 0.332744,
                   0.496631, 0.003346, 0.000023]
        }, ['F0', 'F1', 'F2', 'F3', 'F4', 'X0', 'X1', 'X2']).T

        exp_md = pd.DataFrame(
            {'S0': [0.000000, 5, 10000, 1.0, 3],
             'S1': [0.555556, 5, 10000, 1.0, 3],
             'S2': [1.111111, 5, 10000, 1.0, 3],
             'S3': [1.666667, 5, 10000, 1.0, 3],
             'S4': [2.222222, 5, 10000, 1.0, 3],
             'S5': [2.777778, 5, 10000, 1.0, 3],
             'S6': [3.333333, 5, 10000, 1.0, 3],
             'S7': [3.888889, 5, 10000, 1.0, 3],
             'S8': [4.444444, 5, 10000, 1.0, 3],
             'S9': [5.000000, 5, 10000, 1.0, 3]},
            index=['gradient', 'n_diff', 'library_size',
                   'effect_size', 'n_contaminants']).T
        md = md.astype(np.float)
        md = md.reindex(columns=['gradient', 'n_diff', 'library_size',
                                 'effect_size', 'n_contaminants'])
        pdt.assert_frame_equal(md, exp_md)

        exp_truth = ['F0', 'F1', 'F2', 'F3', 'F4']

        self.assertListEqual(truth, exp_truth)

    def test_compositional_regression_effect_size_generator(self):
        gen = compositional_regression_effect_size_generator(
            max_gradient=5,
            gradient_intervals=10,
            sigma=1,
            n_species=5,
            n_contaminants=2,
            lam=0.1,
            max_beta=1,
            beta_intervals=3
        )
        # second table
        table, md, truth = next(gen)

        exp_table = pd.DataFrame(
            {'S0': [0.1, 0.1, 0.1, 0.1, 0.1, 0.499977, 0.000023],
             'S1': [0.1, 0.1, 0.1, 0.1, 0.1, 0.499977, 0.000023],
             'S2': [0.1, 0.1, 0.1, 0.1, 0.1, 0.499977, 0.000023],
             'S3': [0.1, 0.1, 0.1, 0.1, 0.1, 0.499977, 0.000023],
             'S4': [0.1, 0.1, 0.1, 0.1, 0.1, 0.499977, 0.000023],
             'S5': [0.1, 0.1, 0.1, 0.1, 0.1, 0.499977, 0.000023],
             'S6': [0.1, 0.1, 0.1, 0.1, 0.1, 0.499977, 0.000023],
             'S7': [0.1, 0.1, 0.1, 0.1, 0.1, 0.499977, 0.000023],
             'S8': [0.1, 0.1, 0.1, 0.1, 0.1, 0.499977, 0.000023],
             'S9': [0.1, 0.1, 0.1, 0.1, 0.1, 0.499977, 0.000023]},
            index=['F0', 'F1', 'F2', 'F3', 'F4', 'X0', 'X1']).T
        npt.assert_allclose(table.values, exp_table.values, atol=1e-3)

        exp_md = pd.DataFrame(
            {'S0': [0.000000, 5, 10000, 2, 0],
             'S1': [0.555556, 5, 10000, 2, 0],
             'S2': [1.111111, 5, 10000, 2, 0],
             'S3': [1.666667, 5, 10000, 2, 0],
             'S4': [2.222222, 5, 10000, 2, 0],
             'S5': [2.777778, 5, 10000, 2, 0],
             'S6': [3.333333, 5, 10000, 2, 0],
             'S7': [3.888889, 5, 10000, 2, 0],
             'S8': [4.444444, 5, 10000, 2, 0],
             'S9': [5.000000, 5, 10000, 2, 0]
            }, index=['gradient', 'n_diff', 'library_size',
                      'n_contaminants', 'effect_size']
        ).T
        md = md.astype(np.float)
        md = md.reindex(columns=['gradient', 'n_diff', 'library_size',
                                 'n_contaminants', 'effect_size'])

        pdt.assert_frame_equal(md, exp_md)

        exp_truth = ['F0', 'F1', 'F2', 'F3', 'F4']
        self.assertListEqual(truth, exp_truth)

        # second table
        table, md, truth = next(gen)

        exp_table = pd.DataFrame({
            'S0': [0.200259, 0.164728, 0.091685, 0.034529, 0.008799,
                   0.499977, 0.000023],
            'S1': [0.133485, 0.155384, 0.122386, 0.065225, 0.023521,
                   0.499977, 0.000023],
            'S2': [0.076048, 0.125273, 0.139631, 0.105308, 0.053739,
                   0.499977, 0.000023],
            'S3': [0.036321, 0.084668, 0.133549, 0.142533, 0.102930,
                   0.499977, 0.000023],
            'S4': [0.014642, 0.048301, 0.107814, 0.162835, 0.166407,
                   0.499977, 0.000023],
            'S5': [0.005128, 0.023938, 0.075613, 0.161608, 0.233714,
                   0.499977, 0.000023],
            'S6': [0.001617, 0.010681, 0.047746, 0.144412, 0.295544,
                   0.499977, 0.000023],
            'S7': [0.000474, 0.004428, 0.028010, 0.119886, 0.347203,
                   0.499977, 0.000023],
            'S8': [0.000132, 0.001745, 0.015624, 0.094638, 0.387860,
                   0.499977, 0.000023],
            'S9': [0.000036, 0.000665, 0.008422, 0.072191, 0.418687,
                   0.499977, 0.000023]
        }, index=['F0', 'F1', 'F2', 'F3', 'F4', 'X0', 'X1']).T

        npt.assert_allclose(table.values, exp_table.values, atol=1e-3)

        exp_md = pd.DataFrame(
            {'S0': [0.000000, 5, 10000, 2, 0.5],
             'S1': [0.555556, 5, 10000, 2, 0.5],
             'S2': [1.111111, 5, 10000, 2, 0.5],
             'S3': [1.666667, 5, 10000, 2, 0.5],
             'S4': [2.222222, 5, 10000, 2, 0.5],
             'S5': [2.777778, 5, 10000, 2, 0.5],
             'S6': [3.333333, 5, 10000, 2, 0.5],
             'S7': [3.888889, 5, 10000, 2, 0.5],
             'S8': [4.444444, 5, 10000, 2, 0.5],
             'S9': [5.000000, 5, 10000, 2, 0.5]
            }, index=['gradient', 'n_diff', 'library_size',
                      'n_contaminants', 'effect_size']
        ).T
        md = md.astype(np.float)
        md = md.reindex(columns=['gradient', 'n_diff', 'library_size',
                                 'n_contaminants', 'effect_size'])

        pdt.assert_frame_equal(md, exp_md)

        exp_truth = ['F0', 'F1', 'F2', 'F3', 'F4']
        self.assertListEqual(truth, exp_truth)


if __name__ == "__main__":
    unittest.main()

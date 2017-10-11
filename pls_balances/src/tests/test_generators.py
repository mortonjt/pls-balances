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
    compositional_regression_effect_size_generator,
    library_size_difference_generator)


class TestCompositionalEffectSize(unittest.TestCase):

    def test_composition_effect_size_simple(self):

        gen = compositional_effect_size_generator(max_alpha=1, reps=5,
                                                  intervals=2, n_species=5, n_diff=1,
                                                  fold_balance=False,
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
             'effect_size': [1] * 10,
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
             'effect_size': [10] * 10,
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


    def test_composition_effect_template(self):
        # test template
        np.random.seed(0)
        gen = compositional_effect_size_generator(
            max_alpha=1, reps=5,
            intervals=2, n_species=5, n_diff=1,
            n_contaminants=2, lam=0.1,
            fold_balance=False,
            template=np.array([7.0, 3.0, 1.0, 1.0, 2.0, 4.0, 6.0, 1.0, 10.0]))
        table, metadata, truth = next(gen)
        table, metadata, truth = next(gen)
        exp_table = pd.DataFrame(
            np.array([
                [0.227273, 0.045455, 0.022727, 0.079545,
                 0.125, 0.499977, 0.000023],
                [0.227273, 0.045455, 0.022727, 0.079545,
                 0.125, 0.499977, 0.000023],
                [0.227273, 0.045455, 0.022727, 0.079545,
                 0.125, 0.499977, 0.000023],
                [0.227273, 0.045455, 0.022727, 0.079545,
                 0.125, 0.499977, 0.000023],
                [0.227273, 0.045455, 0.022727, 0.079545,
                 0.125, 0.499977, 0.000023],
                [0.008000, 0.016000, 0.008000, 0.028000,
                 0.440, 0.499977, 0.000023],
                [0.008000, 0.016000, 0.008000, 0.028000,
                 0.440, 0.499977, 0.000023],
                [0.008000, 0.016000, 0.008000, 0.028000,
                 0.440, 0.499977, 0.000023],
                [0.008000, 0.016000, 0.008000, 0.028000,
                 0.440, 0.499977, 0.000023],
                [0.008000, 0.016000, 0.008000, 0.028000,
                 0.440, 0.499977, 0.000023]]),
            index = ['S0', 'S1', 'S2', 'S3', 'S4',
                     'S5', 'S6', 'S7', 'S8', 'S9'],
            columns = ['F0', 'F1', 'F2', 'F3', 'F4', 'X0', 'X1']
        )
        npt.assert_allclose(table.values, exp_table.values,
                           atol=1e-2, rtol=1e-2)


    def test_composition_effect_size_exponential(self):

        gen = library_size_difference_generator(
            reps=4,
            n_species=8,
            n_diff=2,
            lam_diff=0.1,
            n_contaminants=2,
            lam_contaminants=0.1,
            effect_size=2,
            min_library_size=10,
            max_library_size=100,
            intervals=3)

        table, metadata, truth = next(gen)
        table, metadata, truth = next(gen)
        exp_table = pd.DataFrame(
            {'S0': [0.740716, 0.0000336284, 0.037036, 0.037036, 0.037036,
                    0.037036, 0.037036, 0.037036, 0.037034, 0.000002],
             'S1': [0.740716, 0.0000336284, 0.037036, 0.037036, 0.037036,
                    0.037036, 0.037036, 0.037036, 0.037034, 0.000002],
             'S2': [0.740716, 0.0000336284, 0.037036, 0.037036, 0.037036,
                    0.037036, 0.037036, 0.037036, 0.037034, 0.000002],
             'S3': [0.740716, 0.0000336284, 0.037036, 0.037036, 0.037036,
                    0.037036, 0.037036, 0.037036, 0.037034, 0.000002],
             'S4': [0.037036, 0.037036, 0.037036, 0.037036, 0.037036,
                    0.037036, 0.740716, 0.0000336284, 0.037034, 0.000002],
             'S5': [0.037036, 0.037036, 0.037036, 0.037036, 0.037036,
                    0.037036, 0.740716, 0.0000336284, 0.037034, 0.000002],
             'S6': [0.037036, 0.037036, 0.037036, 0.037036, 0.037036,
                    0.037036, 0.740716, 0.0000336284, 0.037034, 0.000002],
             'S7': [0.037036, 0.037036, 0.037036, 0.037036, 0.037036,
                    0.037036, 0.740716, 0.0000336284, 0.037034, 0.000002]
            }, index=['F0', 'F1', 'F2', 'F3', 'F4',
                      'F5', 'F6', 'F7', 'X0', 'X1']).T

        pdt.assert_frame_equal(table, exp_table, check_less_precise=True)

        exp_metadata = pd.DataFrame(
            {'S0': [0, 4, 2, 10],
             'S1': [0, 4, 2, 55],
             'S2': [0, 4, 2, 10],
             'S3': [0, 4, 2, 55],
             'S4': [1, 4, 2, 10],
             'S5': [1, 4, 2, 55],
             'S6': [1, 4, 2, 10],
             'S7': [1, 4, 2, 55]},
            index=['group', 'n_diff', 'effect_size', 'library_size']).T
        pdt.assert_frame_equal(metadata, exp_metadata, check_less_precise=True)
        exp_truth = ['F0', 'F1', 'F6', 'F7']
        self.assertListEqual(truth, exp_truth)

    def test_composition_effect_size_balanced(self):

        gen = compositional_effect_size_generator(max_alpha=1, reps=5,
                                                  intervals=2, n_species=5, n_diff=1,
                                                  n_contaminants=2, lam=0.1,
                                                  fold_balance=True, template=None)
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
             'effect_size': [10] * 10,
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
            fold_balance=False,
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

        exp_truth = ['F0', 'F1', 'F3', 'F4']
        self.assertListEqual(truth, exp_truth)

    def test_composition_variable_features_template(self):
        # test template
        np.random.seed(0)

        gen = compositional_variable_features_generator(
            max_changing=2, fold_change=2, reps=5,
            intervals=2, n_species=5,
            fold_balance=False,
            n_contaminants=2, lam=0.1,
            template=np.array([7.0, 3.0, 1.0, 1.0, 2.0, 4.0, 6.0, 1.0, 10.0]))

        table, metadata, truth = next(gen)
        table, metadata, truth = next(gen)

        exp_table = pd.DataFrame([
            [0.062500, 0.125000, 0.031250, 0.109375,
             0.171875, 0.499977, 0.000023],
            [0.062500, 0.125000, 0.031250, 0.109375,
             0.171875, 0.499977, 0.000023],
            [0.062500, 0.125000, 0.031250, 0.109375,
             0.171875, 0.499977, 0.000023],
            [0.062500, 0.125000, 0.031250, 0.109375,
             0.171875, 0.499977, 0.000023],
            [0.062500, 0.125000, 0.031250, 0.109375,
             0.171875, 0.499977, 0.000023],
            [0.022727, 0.045455, 0.022727, 0.159091,
             0.250000, 0.499977, 0.000023],
            [0.022727, 0.045455, 0.022727, 0.159091,
             0.250000, 0.499977, 0.000023],
            [0.022727, 0.045455, 0.022727, 0.159091,
             0.250000, 0.499977, 0.000023],
            [0.022727, 0.045455, 0.022727, 0.159091,
             0.250000, 0.499977, 0.000023],
            [0.022727, 0.045455, 0.022727, 0.159091,
             0.250000, 0.499977, 0.000023]],
            index = ['S0', 'S1', 'S2', 'S3', 'S4',
                     'S5', 'S6', 'S7', 'S8', 'S9'],
            columns = ['F0', 'F1', 'F2', 'F3', 'F4', 'X0', 'X1']
        )

        npt.assert_allclose(table.values, exp_table.values, atol=1e-3, rtol=1e-3)

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

        exp_truth = ['F0', 'F1', 'F3', 'F4']
        self.assertListEqual(truth, exp_truth)

    def test_composition_variable_features_balanced(self):
        gen = compositional_variable_features_generator(
            max_changing=2, fold_change=2, reps=5,
            intervals=2, n_species=5,
            fold_balance=True,
            n_contaminants=2, lam=0.1)

        table, metadata, truth = next(gen)
        table, metadata, truth = next(gen)

        exp_table = pd.DataFrame(
            [[0.100000, 0.100000, 0.100000, 0.100000,
              0.100000, 0.499977, 0.000023],
             [0.100000, 0.100000, 0.100000, 0.100000,
              0.100000, 0.499977, 0.000023],
             [0.100000, 0.100000, 0.100000, 0.100000,
              0.100000, 0.499977, 0.000023],
             [0.100000, 0.100000, 0.100000, 0.100000,
              0.100000, 0.499977, 0.000023],
             [0.100000, 0.100000, 0.100000, 0.100000,
              0.100000, 0.499977, 0.000023],
             [0.041667, 0.041667, 0.083333, 0.166667,
              0.166667, 0.499977, 0.000023],
             [0.041667, 0.041667, 0.083333, 0.166667,
              0.166667, 0.499977, 0.000023],
             [0.041667, 0.041667, 0.083333, 0.166667,
              0.166667, 0.499977, 0.000023],
             [0.041667, 0.041667, 0.083333, 0.166667,
              0.166667, 0.499977, 0.000023],
             [0.041667, 0.041667, 0.083333, 0.166667,
              0.166667, 0.499977, 0.000023]],
            index = ['S0', 'S1', 'S2', 'S3', 'S4',
                     'S5', 'S6', 'S7', 'S8', 'S9'],
            columns = ['F0', 'F1', 'F2', 'F3', 'F4', 'X0', 'X1']
        )

        npt.assert_allclose(table.values, exp_table.values, atol=1e-3, rtol=1e-3)

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

        exp_truth = ['F0', 'F1', 'F3', 'F4']
        self.assertListEqual(truth, exp_truth)

    def test_composition_asymmetric_variable_features(self):
        gen = compositional_variable_features_generator(
            max_changing=2, fold_change=2, reps=5,
            intervals=2, n_species=5, asymmetry=True,
            fold_balance=False,
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
                    np.array([0.1]*5 + [0.499977, 0.00002269]),
                    np.array([0.1]*5 + [0.499977, 0.00002269]),
                    np.array([0.1]*5 + [0.499977, 0.00002269]),
                    np.array([0.1]*5 + [0.499977, 0.00002269]),
                    np.array([0.1]*5 + [0.499977, 0.00002269])
                ))),
            index = ['S0', 'S1', 'S2', 'S3', 'S4',
                     'S5', 'S6', 'S7', 'S8', 'S9'],
            columns = ['F0', 'F1', 'F2', 'F3', 'F4', 'X0', 'X1']
        )

        pdt.assert_frame_equal(table, exp_table, check_less_precise=True)

        exp_metadata = pd.DataFrame(
            {'group': [0] * 5 + [1] * 5,
             'n_diff': [2] * 10,
             'effect_size': [2] * 10,
             'library_size': [10000] * 10
            },
            index = ['S0', 'S1', 'S2', 'S3', 'S4',
                     'S5', 'S6', 'S7', 'S8', 'S9'],
        )

        metadata = metadata.reindex_axis(sorted(metadata.columns), axis=1)
        exp_metadata = exp_metadata.reindex_axis(sorted(exp_metadata.columns), axis=1)
        pdt.assert_frame_equal(metadata, exp_metadata)

        exp_truth = ['F0', 'F1']
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
        table, md, truth = next(gen)
        # second table
        table, md, truth = next(gen)

        exp_table = pd.DataFrame(
            {'S0': [0.200259, 0.164728, 0.091685, 0.034529, 0.008799,
                    0.499977, 0.000023],
             'S1': [0.166407, 0.162835, 0.107814, 0.048301, 0.014642,
                    0.499977, 0.000023],
             'S2': [0.133485, 0.155384, 0.122386, 0.065225, 0.023521,
                    0.499977, 0.000023],
             'S3': [0.102930, 0.142533, 0.133549, 0.084668, 0.036321,
                    0.499977, 0.000023],
             'S4': [0.076048, 0.125273, 0.139631, 0.105308, 0.053739,
                    0.499977, 0.000023],
             'S5': [0.053739, 0.105308, 0.139631, 0.125273, 0.076048,
                    0.499977, 0.000023],
             'S6': [0.036321, 0.084668, 0.133549, 0.142533, 0.102930,
                    0.499977, 0.000023],
             'S7': [0.023521, 0.065225, 0.122386, 0.155384, 0.133485,
                    0.499977, 0.000023],
             'S8': [0.014642, 0.048301, 0.107814, 0.162835, 0.166407,
                    0.499977, 0.000023],
             'S9': [0.008799, 0.034529, 0.091685, 0.164728, 0.200259,
                    0.499977, 0.000023]},
            index=['F0', 'F1', 'F2', 'F3', 'F4', 'X0', 'X1']).T
        npt.assert_allclose(table.values, exp_table.values, atol=1e-3)

        exp_md = pd.DataFrame({
            'S0':  [0.000000, 5.0, 10000.0, 2.0, 1.0],
            'S1':  [0.277778, 5.0, 10000.0, 2.0, 1.0],
            'S2':  [0.555556, 5.0, 10000.0, 2.0, 1.0],
            'S3':  [0.833333, 5.0, 10000.0, 2.0, 1.0],
            'S4':  [1.111111, 5.0, 10000.0, 2.0, 1.0],
            'S5':  [1.388889, 5.0, 10000.0, 2.0, 1.0],
            'S6':  [1.666667, 5.0, 10000.0, 2.0, 1.0],
            'S7':  [1.944444, 5.0, 10000.0, 2.0, 1.0],
            'S8':  [2.222222, 5.0, 10000.0, 2.0, 1.0],
            'S9':  [2.500000, 5.0, 10000.0, 2.0, 1.0]
        }, index=['gradient', 'n_diff', 'library_size',
                  'n_contaminants', 'effect_size']).T
        md = md.astype(np.float)
        md = md.reindex(columns=['gradient', 'n_diff', 'library_size',
                                 'n_contaminants', 'effect_size'])
        pdt.assert_frame_equal(md, exp_md)

        exp_truth = ['F0', 'F1', 'F2', 'F3', 'F4']
        self.assertListEqual(truth, exp_truth)


if __name__ == "__main__":
    unittest.main()

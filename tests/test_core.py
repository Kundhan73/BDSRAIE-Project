import unittest
import numpy as np
import pandas as pd
from fairness_core import count_metrics, reweighing_table, threshold_probability, ScoreAdapter


class CountTests(unittest.TestCase):
    def test_metrics_use_repay_as_positive(self):
        rows = pd.DataFrame([
            {'group': 0, 'tn': 30, 'fp': 10, 'fn': 20, 'tp': 40},
            {'group': 1, 'tn': 40, 'fp': 10, 'fn': 10, 'tp': 40},
        ])
        overall, groups = count_metrics(rows)
        self.assertAlmostEqual(overall['accuracy'], .75)
        self.assertAlmostEqual(overall['equal_opportunity_gap'], .8-2/3)
        self.assertAlmostEqual(overall['equalized_odds_gap'], .8-2/3)
        self.assertAlmostEqual(groups[0]['false_positive_rate'], .25)

    def test_zero_denominator_is_undefined(self):
        overall, groups = count_metrics(pd.DataFrame([
            {'group': 0, 'tn': 10, 'fp': 0, 'fn': 0, 'tp': 0},
            {'group': 1, 'tn': 8, 'fp': 0, 'fn': 0, 'tp': 0}]))
        self.assertTrue(np.isnan(overall['equal_opportunity_gap']))

    def test_reweighing_preserves_total_and_removes_label_association(self):
        cells = pd.DataFrame({'group':[0,0,1,1], 'label':[0,1,0,1], 'count':[10,30,30,30]})
        w = reweighing_table(cells)
        w['mass'] = w['count']*w['weight']
        self.assertAlmostEqual(w.mass.sum(), 100)
        rates = w.groupby('group').apply(lambda x: x.loc[x.label==1,'mass'].sum()/x.mass.sum(), include_groups=False)
        np.testing.assert_allclose(rates, [.6,.6])

    def test_threshold_mixture_handles_boundaries(self):
        rule = {'p0':.25,'p1':.75,'operation0':{'operator':'>','threshold':.4},
                'operation1':{'operator':'>','threshold':.7}}
        np.testing.assert_allclose(threshold_probability(np.array([.3,.4,.5,.7,.8]),rule), [0,0,.25,.25,1])

    def test_score_adapter_exposes_repay_probabilities(self):
        a=ScoreAdapter()
        np.testing.assert_allclose(a.predict_proba([[.2],[.9]]), [[.8,.2],[.1,.9]])


if __name__ == '__main__':
    unittest.main()

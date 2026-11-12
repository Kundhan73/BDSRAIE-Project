"""Small, independently checked numerical parts of the distributed audit."""
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin


class ScoreAdapter(ClassifierMixin, BaseEstimator):
    """Expose already fitted Spark repayment scores through the sklearn API."""
    def __init__(self):
        self.classes_ = np.array([0, 1])
        self.fitted_ = True

    def fit(self, X, y=None):
        return self

    def predict_proba(self, X):
        p = np.asarray(X, dtype=float).reshape(-1)
        return np.column_stack((1-p, p))

    def predict(self, X):
        return (self.predict_proba(X)[:, 1] > .5).astype(int)


def reweighing_table(cells):
    """Kamiran-Calders group-label independence weights, fitted on train only."""
    w = cells.copy()
    n = w['count'].sum()
    group_n = w.groupby('group')['count'].transform('sum')
    label_n = w.groupby('label')['count'].transform('sum')
    if (w['count'] <= 0).any() or len(w) != 4:
        raise ValueError('Both labels must occur in both audit groups')
    w['weight'] = group_n * label_n / (n*w['count'])
    return w


def threshold_probability(scores, rule):
    scores = np.asarray(scores)
    def op(spec):
        return (scores > spec['threshold']) if spec['operator'] == '>' else (scores < spec['threshold'])
    return rule['p0']*op(rule['operation0'])+rule['p1']*op(rule['operation1'])


def count_metrics(counts):
    def div(a,b):
        return float(a/b) if b else float('nan')
    def rates(row):
        tn, fp, fn, tp = [int(row[k]) for k in ('tn','fp','fn','tp')]
        n=tn+fp+fn+tp
        return {'n':n, 'tn':tn,'fp':fp,'fn':fn,'tp':tp,
                'accuracy':div(tp+tn,n), 'selection_rate':div(tp+fp,n),
                'true_positive_rate':div(tp,tp+fn), 'false_positive_rate':div(fp,fp+tn),
                'precision':div(tp,tp+fp), 'repayment_rate':div(tp+fn,n),
                'approved_default_rate':div(fp,tp+fp)}
    groups=[dict(group=int(r['group']),**rates(r)) for r in counts.to_dict('records')]
    total=rates(counts[['tn','fp','fn','tp']].sum())
    def gap(k):
        a=np.array([g[k] for g in groups]); return float(a.max()-a.min())
    total['balanced_accuracy']=(total['true_positive_rate']+1-total['false_positive_rate'])/2
    total['demographic_parity_gap']=gap('selection_rate')
    total['equal_opportunity_gap']=gap('true_positive_rate')
    total['equalized_odds_gap']=float(np.maximum(gap('true_positive_rate'),gap('false_positive_rate')))
    s=[g['selection_rate'] for g in groups]
    total['demographic_parity_ratio']=div(min(s),max(s))
    return total,groups

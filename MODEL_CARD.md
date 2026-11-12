# Model card: synthetic lending repayment model

**Developer:** Kundhan (PES1UG24AM327)  
**Version/date:** 1.0 / 28 October 2026  
**Status:** Academic experiment; not approved for real lending.  
**Selected decision policy:** `to_tol0.010`  
**Model family:** Spark ML binary logistic regression with a repayment score.

## Intended use and users

Demonstrate scalable fairness assessment and mitigation for a Big Data case
study. Intended users are the student, evaluator and technical reviewers. Do
not use for live underwriting, pricing, credit limits or eligibility decisions.
No real customers or Home Credit records are used.

## Inputs and outputs

Inputs: income_z, debt_z, history_measured and employment_z, in the standardized
units defined by the generator. These are synthetic descriptors, not a schema
for a production credit bureau. Output score: probability-like repayment score;
it has not received a calibration audit. Decision 1 means approve; label 1
means would repay. Group is excluded from logistic features but used for audit,
training reweighing and Fairlearn group-dependent threshold policies. Missing
features, unseen groups and production data schemas are outside scope.

## Data, training and evaluation

2,000,000 synthetic applications with seed 327. Train:
1,200,876; validation: 399,890; test:
399,234. Hash splits are disjoint. Group 1 prevalence is about
45%. The generator reduces measured history by 0.9 for group 1 but uses true
history for repayment labels. Groups also have different input distributions.
Labels are available for all applications, unlike many real lending datasets.

Three logistic models use L2 regularization 0.01, at most 40 iterations and
weight strengths 0, 0.5 and 1.0. ThresholdOptimizer fits on a bounded validation
sample of 96,021 scores. 41 policies
are evaluated. Selection maximizes validation accuracy with TPR gap <=0.02 and
approval between 0.2 and 0.9, with a documented minimum-gap fallback.
Fallback used: False. Test results never enter selection.

## Test results

| Metric | Baseline | Selected |
|---|---:|---:|
| Accuracy | 77.49% | 78.02% |
| Balanced accuracy | 74.80% | 75.55% |
| Equal opportunity gap | 18.29 pp | 0.73 pp |
| Equalized odds gap | 30.22 pp | 1.14 pp |
| Demographic parity gap | 27.94 pp | 6.33 pp |
| Demographic parity ratio | 0.649 | 0.909 |
| Approval rate | 67.16% | 66.44% |
| False positive rate | 36.41% | 34.77% |
| Default among approved | 20.61% | 19.89% |

Equal opportunity is the main applicant-facing metric in this case. A smaller
gap means more similar approval rates among would-repay applicants. It does
not establish fairness across every error type or prove legal compliance.

## Selected policy configuration

```json
{
  "policy": "to_tol0.010",
  "method": "threshold_optimizer",
  "tolerance": 0.01,
  "rules": {
    "0": {
      "p0": 0.5891666666667431,
      "p1": 0.4108333333332569,
      "operation0": {
        "operator": ">",
        "threshold": 0.6216048360329043
      },
      "operation1": {
        "operator": ">",
        "threshold": 0.6206174701372649
      }
    },
    "1": {
      "p0": 0.708518427518428,
      "p1": 0.291481572481572,
      "operation0": {
        "operator": ">",
        "threshold": 0.40607887076499416
      },
      "operation1": {
        "operator": ">",
        "threshold": 0.36621391885204974
      }
    }
  }
}
```

Fairlearn postprocessing can randomize between thresholds. The supplied
pipeline uses a stable ID hash (seed salt 900) to realize those probabilities
in Spark. Re-evaluated outcomes can differ from calibration constraints.
Group membership is required at inference if the chosen policy is a Fairlearn
threshold policy. The actual chosen method is `threshold_optimizer`.

## Limitations and ethical considerations

Synthetic results cannot establish real-world accuracy or harm reduction.
Only one fictional binary group attribute, one seed and one static population
are tested. Intersectional and small groups, calibration, ranking metrics,
individual fairness, temporal drift and uncertainty from retraining are not
evaluated. Conditional test-rate intervals do not cover those uncertainties.
Group-sensitive decisions and randomization need review before any real use.
The 2-point fairness target has no claimed legal status. Historical labels and
outcome selection would need an independent audit on real data.

## Monitoring and release conditions

No production release is recommended. Before a real pilot, establish allowed
uses of protected data, representative time-based validation, affordability
and loss constraints, calibration by group, outcome observation rules, appeal
and human review procedures, and approval by relevant domain reviewers. After
deployment, monitor approval, TPR/FPR with mature outcomes, default among
approved, subgroup sample sizes, calibration and drift; log model/policy
versions and decision seeds. Define operational thresholds before rollout.

## Reproducibility and ownership

Spark 3.5.3; Fairlearn 0.13.0; Python
3.11.16. Full audit elapsed 19.6 seconds on
local[4] with 8 partitions. Maximum Spark/Fairlearn
metric error: 0. Export probability error:
0. See requirements.txt, audit.py,
results/run_manifest.json, results/model_coefficients.json and
results/policies.json. Contact/owner for this academic artifact: Kundhan.
Model card structure follows Mitchell et al. (2019),
https://arxiv.org/abs/1810.03993.

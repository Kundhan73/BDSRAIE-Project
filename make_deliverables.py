"""Build all submission artifacts from the completed, measured audit outputs."""
import json
import math
import os
from pathlib import Path
import shutil
import textwrap
import zipfile
from xml.sax.saxutils import escape

os.environ.setdefault('MPLCONFIGDIR',str(Path('work/mpl').resolve()))
os.environ.setdefault('XDG_CACHE_HOME',str(Path('work/cache').resolve()))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.enums import TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image, PageBreak

ROOT=Path(__file__).resolve().parent
RESULTS=ROOT/'results'
OUT=ROOT/'outputs'
PLOTS=ROOT/'plots'
OUT.mkdir(exist_ok=True); PLOTS.mkdir(exist_ok=True)
M=json.loads((RESULTS/'run_manifest.json').read_text())
S=pd.read_csv(RESULTS/'metrics.csv')
G=pd.read_csv(RESULTS/'group_metrics.csv')
D=pd.read_csv(RESULTS/'data_profile.csv')
P=json.loads((RESULTS/'policies.json').read_text())
TEST=S[S.split=='test'].set_index('policy')
VAL=S[S.split=='validation'].set_index('policy')
BASE=M['representatives']['baseline']; RW=M['representatives']['reweighing']
TO=M['representatives']['threshold_optimizer']; PICK=M['selected_policy']
B=TEST.loc[BASE]; C=TEST.loc[PICK]
NAMES={'baseline':'Baseline','reweighing':'Reweighing','threshold_optimizer':'Fairlearn thresholds'}
COLORS={'baseline':'#526c91','reweighing':'#c27c29','threshold_optimizer':'#198574'}
INK=colors.HexColor('#17324d'); TEAL=colors.HexColor('#198574')
styles=getSampleStyleSheet()
styles.add(ParagraphStyle(name='TitleCustom',fontName='Helvetica-Bold',fontSize=27,leading=32,textColor=INK,spaceAfter=14))
styles.add(ParagraphStyle(name='SubCustom',fontName='Helvetica',fontSize=12,leading=17,textColor=colors.HexColor('#526477'),spaceAfter=15))
styles.add(ParagraphStyle(name='BodyCustom',fontName='Helvetica',fontSize=10,leading=14.5,spaceAfter=9,textColor=colors.HexColor('#243443')))
styles.add(ParagraphStyle(name='SmallCustom',fontName='Helvetica',fontSize=8.2,leading=11,spaceAfter=7,textColor=colors.HexColor('#526477')))
styles.add(ParagraphStyle(name='HCustom',fontName='Helvetica-Bold',fontSize=15,leading=20,textColor=INK,spaceBefore=8,spaceAfter=10))
styles.add(ParagraphStyle(name='TableCustom',fontName='Helvetica',fontSize=8.3,leading=11,textColor=colors.HexColor('#243443')))
styles.add(ParagraphStyle(name='THCustom',fontName='Helvetica-Bold',fontSize=8.2,leading=11,textColor=colors.white))
styles.add(ParagraphStyle(name='CodeCustom',fontName='Courier',fontSize=8,leading=11,spaceAfter=8,textColor=INK))
W=A4[0]-100


def pct(x): return f'{100*x:.2f}%'
def pp(x): return f'{100*x:.2f} pp'
def para(t,style='BodyCustom'): return Paragraph(t,styles[style])
def heading(t): return para(t,'HCustom')
def small(t): return para(t,'SmallCustom')
def table(headers,rows,widths=None):
    data=[[para(escape(str(x)),'THCustom') for x in headers]]
    data += [[para(escape(str(x)),'TableCustom') for x in r] for r in rows]
    t=Table(data,colWidths=widths,repeatRows=1,hAlign='LEFT')
    t.setStyle(TableStyle([('BACKGROUND',(0,0),(-1,0),INK),('VALIGN',(0,0),(-1,-1),'TOP'),
       ('ROWBACKGROUNDS',(0,1),(-1,-1),[colors.HexColor('#f1f5f8'),colors.white]),
       ('LEFTPADDING',(0,0),(-1,-1),8),('RIGHTPADDING',(0,0),(-1,-1),8),
       ('TOPPADDING',(0,0),(-1,-1),8),('BOTTOMPADDING',(0,0),(-1,-1),8),
       ('LINEBELOW',(0,-1),(-1,-1),.5,colors.HexColor('#cbd5df'))]))
    return t


def footer(canvas,doc):
    canvas.saveState();canvas.setStrokeColor(colors.HexColor('#dce3e9'))
    canvas.line(50,43,A4[0]-50,43);canvas.setFont('Helvetica',8)
    canvas.setFillColor(colors.HexColor('#637487'))
    canvas.drawString(50,29,'Kundhan | Credit Scoring Fairness Audit | Synthetic data')
    canvas.drawRightString(A4[0]-50,29,str(doc.page));canvas.restoreState()


def build_pdf(path,story,title):
    doc=SimpleDocTemplate(str(path),pagesize=A4,rightMargin=50,leftMargin=50,topMargin=45,bottomMargin=57,
                          title=title,author='Kundhan',subject='Big Data Project - synthetic lending fairness audit')
    doc.build(story,onFirstPage=footer,onLaterPages=footer)


def save_fig(fig,name):
    fig.savefig(PLOTS/f'{name}.png',dpi=220,bbox_inches='tight',facecolor='white')
    fig.savefig(PLOTS/f'{name}.pdf',bbox_inches='tight',facecolor='white')
    plt.close(fig)


def charts():
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.spines.top':False,
                         'axes.spines.right':False,'axes.labelcolor':'#243443','axes.titleweight':'bold'})
    fig,axes=plt.subplots(1,2,figsize=(11.5,4.5))
    for ax,metric,title in zip(axes,['equal_opportunity_gap','equalized_odds_gap'],['Equal opportunity','Equalized odds']):
        for method in NAMES:
            q=TEST[TEST.method==method]
            ax.scatter(q[metric]*100,q.accuracy*100,s=40,alpha=.65,label=NAMES[method],color=COLORS[method],edgecolors='white',linewidths=.5)
        ax.scatter(C[metric]*100,C.accuracy*100,marker='*',s=230,color='#172d43',label='Selected on validation',zorder=5)
        ax.scatter(B[metric]*100,B.accuracy*100,marker='s',s=100,facecolors='none',edgecolors='#172d43',linewidths=1.5,zorder=4)
        ax.set(xlabel=f'{title} gap (percentage points; lower is better)',ylabel='Test accuracy (%)',title=title)
        ax.grid(alpha=.15)
    axes[0].legend(fontsize=8,loc='lower left',frameon=True)
    fig.suptitle(f'Accuracy and fairness across {M["policy_count"]} decision policies',fontsize=15,y=1.03)
    fig.tight_layout();save_fig(fig,'accuracy_fairness_tradeoff')
    fig,axes=plt.subplots(1,2,figsize=(10.5,3.7))
    for ax,metric,title in zip(axes,['true_positive_rate','false_positive_rate'],['Approval among would-repay applicants (TPR)','Approval among would-default applicants (FPR)']):
        x=np.arange(2)
        for j,(policy,label,color) in enumerate([(BASE,'Baseline','#526c91'),(PICK,'Selected','#198574')]):
            q=G[(G.split=='test')&(G.policy==policy)].sort_values('group')
            ax.bar(x+(j-.5)*.31,q[metric]*100,width=.30,label=label,color=color)
        ax.set_xticks(x,['Group 0','Group 1']);ax.set_ylim(0,100);ax.set_ylabel('Rate (%)');ax.set_title(title,fontsize=10);ax.grid(axis='y',alpha=.15);ax.set_axisbelow(True)
    axes[0].legend(fontsize=9);fig.tight_layout();save_fig(fig,'group_error_rates')
    fig,ax=plt.subplots(figsize=(8.8,3.8))
    for method in NAMES:
        q=VAL[VAL.method==method]
        ax.scatter(q.equal_opportunity_gap*100,q.accuracy*100,color=COLORS[method],label=NAMES[method],s=40,alpha=.7)
    q=VAL.loc[PICK];ax.scatter(q.equal_opportunity_gap*100,q.accuracy*100,s=210,marker='*',color='#172d43',zorder=6)
    ax.axvline(2,color='#687789',ls='--',lw=1,label='2 pp validation target')
    ax.set(xlabel='Validation equal opportunity gap (percentage points)',ylabel='Validation accuracy (%)',title='Policy selection used validation data only')
    ax.grid(alpha=.15);ax.legend(fontsize=8);fig.tight_layout();save_fig(fig,'validation_policy_selection')


def comparison_rows():
    rows=[]
    for policy,name in [(BASE,'Baseline'),(RW,'Reweighing'),(TO,'Fairlearn thresholds'),(PICK,'Selected policy')]:
        r=TEST.loc[policy]
        rows.append([name,pct(r.accuracy),pp(r.equal_opportunity_gap),pp(r.equalized_odds_gap),pct(r.approved_default_rate)])
    return rows


def uncertainty():
    rows=[]
    for policy in [BASE,RW,TO,PICK]:
        q=G[(G.split=='test')&(G.policy==policy)].sort_values('group')
        a,b=q.iloc[0],q.iloc[1]
        delta=a.true_positive_rate-b.true_positive_rate
        se=math.sqrt(a.true_positive_rate*(1-a.true_positive_rate)/(a.tp+a.fn)+b.true_positive_rate*(1-b.true_positive_rate)/(b.tp+b.fn))
        rows.append({'policy':policy,'signed_tpr_group0_minus_group1':delta,'ci95_low':delta-1.96*se,'ci95_high':delta+1.96*se})
    result=pd.DataFrame(rows).drop_duplicates('policy');result.to_csv(RESULTS/'uncertainty.csv',index=False)
    return result.set_index('policy')


def report(ci):
    story=[]
    story += [small('BIG DATA PROJECT / 28 OCTOBER 2026'),para('Credit Scoring<br/>Fairness Audit','TitleCustom'),
              para('A Spark and Fairlearn case study with two million synthetic applications','SubCustom'),
              para('Kundhan<br/>SRN: PES1UG24AM327','BodyCustom')]
    story += [heading('What the audit found'),para(
        f'The validation-selected policy achieved {pct(C.accuracy)} test accuracy with an equal opportunity gap of {pp(C.equal_opportunity_gap)}. '
        f'The baseline gap was {pp(B.equal_opportunity_gap)} at {pct(B.accuracy)} accuracy. '
        'I used equal opportunity as the main fairness check because it measures whether applicants who would repay have similar chances of approval across groups.')]
    story += [table(['Test comparison','Accuracy','TPR gap','EOdds gap','Default among approved'],comparison_rows(),[125,72,75,85, W-357]),Spacer(1,10),
       small('Each family representative maximizes validation accuracy within its family. The selected policy also meets the stated validation fairness/approval criteria. A repeated row means the same policy won both comparisons. pp = percentage points.'),
       para(f'The selected policy is <b>{PICK}</b>. Its test accuracy changes by {100*(C.accuracy-B.accuracy):+.2f} percentage points relative to the baseline. '
            f'Its overall approval rate is {pct(C.selection_rate)}, and {pct(C.approved_default_rate)} of approved applicants would default in this synthetic experiment.'),
       para('The main lesson is that a smaller repayment-opportunity gap does not settle every lending concern. The false positive rate and the default rate among approved applicants can move differently, so I report them alongside the fairness target.'),
       small('Scope: the assignment permits a large synthetic equivalent. No Home Credit records or real customer data were used. The experiment demonstrates the pipeline and its trade-offs; it does not validate a bank model.'),PageBreak()]
    story += [heading('1. Data and experimental setup'),para(
       'I generated 2,000,000 independent application IDs directly in Spark. Group 1 accounts for approximately 45% of applications. The two groups are fictional audit groups; they do not represent a particular gender, caste or ethnicity. '
       'Feature shifts and a biased credit-history measurement make the fairness question visible without needing a Kaggle account.'),
       table(['Input','Construction / role'],[
       ['income_z','Standard normal minus 0.35 for group 1'],['debt_z','Standard normal plus 0.25 for group 1'],
       ['history_measured','True history minus 0.90 for group 1'],['employment_z','Standard normal minus 0.15 for group 1'],
       ['group','Binary audit attribute; excluded from logistic model features'],['label','1 = would repay; 0 = would default. Generated for every applicant.']], [120,W-120]),Spacer(1,10),
       para('The repayment probability is sigmoid(1.1 + income_z - 1.2 debt_z + 0.9 history_true + 0.45 employment_z). The outcome is a Bernoulli draw from that probability. '
            'The model sees measured history, while the generator uses true history. This deliberately creates measurement bias. The groups also differ in their feature and repayment distributions.'),
       table(['Split','Applications','Purpose'],[[s,f'{M["splits"][s]:,}',purpose] for s,purpose in [('train','Fit models and reweighing weights'),('validation','Fit thresholds and choose the policy'),('test','Final performance and fairness audit')]], [85,105,W-190]),Spacer(1,10),
       para(f'All splits use an ID hash with seed {M["seed"]}, approximately 60/20/20. Separate hash salts control outcomes, calibration sampling and randomized decisions. '
            'The generator is invariant to partition count. No applicant ID or group variable enters the base model.'),
       para('I fit three Spark logistic models with the same four features, L2 regularization 0.01 and at most 40 iterations. The only training change is reweighing strength. '
            'There is no imputation or external preprocessing to fit because all synthetic inputs are complete numeric values.'),
       small('This dataset is equivalent in scale and task type, not a statistical replica of Home Credit. Its synthetic distributions and known outcomes make it more convenient than a real application dataset, where outcomes for rejected applicants are usually missing.'),PageBreak()]
    story += [heading('2. Metrics and mitigation methods'),para(
       'A decision of 1 means approve. A true positive is an approval for someone who would repay; a false positive is an approval for someone who would default. Fixing this convention matters: using default as the positive label would reverse the lending interpretation.'),
       table(['Metric','Definition','Why report it'],[
       ['Accuracy','(TP + TN) / N','Predictive correctness; can hide group errors'],
       ['Equal opportunity gap','max(TPR_g) - min(TPR_g)','Unequal approval of would-repay applicants'],
       ['Equalized odds gap','max(TPR gap, FPR gap)','Checks both kinds of error disparity'],
       ['Demographic parity gap','max(approval_g) - min(approval_g)','Overall allocation difference'],
       ['DP ratio','min(approval_g) / max(approval_g)','Relative allocation difference'],
       ['Approved-default rate','FP / (TP + FP)','Risk within the approved portfolio']], [110,170,W-280]),Spacer(1,10),
       para('<b>Method 1 - training reweighing.</b> I estimate w(g,y) = N_g N_y / (N N_gy) from training data only [3]. '
            'The final weight is (1 - alpha) + alpha w(g,y), with alpha equal to 0, 0.5 or 1. Spark passes it to logistic regression through weightCol. '
            'At strength 1 the weighted training labels and groups are independent; this does not guarantee equal test error rates.'),
       para('<b>Method 2 - Fairlearn threshold optimization.</b> The already fitted baseline repayment score is passed through a small sklearn-compatible adapter. '
            'ThresholdOptimizer fits group-specific, potentially randomized rules with true_positive_rate_parity and accuracy_score [1,2]. '
            'I compare tolerances 0, 0.01, 0.025, 0.05 and 0.10 on a grid of 1,000 points. These constraints apply to calibration data, so held-out gaps need not match them exactly.'),
       para(f'The calibration sample contains {M["threshold_calibration_n"]:,} validation applications (hard cap 100,000). Exported rules are checked against Fairlearn probabilities and evaluated in Spark. '
            'For each logistic model, common thresholds from 0.30 to 0.85 in steps of 0.05 supply additional accuracy/fairness operating points.'),
       para('The selection rule maximizes full-validation accuracy among policies with a TPR gap of at most 0.02 and an approval rate between 0.2 and 0.9. '
            f'The minimum-gap fallback was {"used" if M["selection_fallback_used"] else "not needed"}. The test split is excluded from this choice. '
            'The 2-point target and approval bounds are transparent experiment choices, not regulatory thresholds.'),PageBreak()]
    story += [heading('3. Accuracy versus fairness'),Image(str(PLOTS/'accuracy_fairness_tradeoff.png'),width=W,height=W*4.5/11.5),
       small(f'Figure 1. All {M["policy_count"]} policies evaluated on {M["splits"]["test"]:,} untouched test applications. The star is the validation-selected policy; the open square is the validation-selected baseline. These are evaluated operating points, not a fitted Pareto frontier.'),
       para(f'The baseline TPR gap is {pp(B.equal_opportunity_gap)}; the selected policy reduces it to {pp(C.equal_opportunity_gap)}. '
            f'The corresponding accuracy change is {100*(C.accuracy-B.accuracy):+.2f} percentage points. '
            f'The equalized odds gap changes from {pp(B.equalized_odds_gap)} to {pp(C.equalized_odds_gap)}. '
            'The two panels matter because a policy optimized for one conditional rate can leave the other conditional rate unequal.'),
       para(f'The best-accuracy Fairlearn family representative has {pct(TEST.loc[TO,"accuracy"])} test accuracy at a {pp(TEST.loc[TO,"equal_opportunity_gap"])} TPR gap. '
            f'The stricter selected policy gives up {100*(TEST.loc[TO,"accuracy"]-C.accuracy):.2f} accuracy points relative to that representative. '
            'Both can outperform the baseline because the deliberately biased history measurement leaves room to improve accuracy and fairness together. '
            f'Reweighing alone changes the gap from {pp(B.equal_opportunity_gap)} to {pp(TEST.loc[RW,"equal_opportunity_gap"])}; it does not directly repair that measurement bias.'),
       Image(str(PLOTS/'validation_policy_selection.png'),width=W,height=W*3.8/8.8),
       small('Figure 2. Validation operating points and the predeclared 2-point TPR-gap target. Only validation results determine the selected policy.'),
       para('I would not choose the point furthest left without checking its approval rate and accuracy. Approving almost everyone or almost no one can make group gaps small while producing a poor lending decision rule. The approval bounds prevent those extremes from winning this experiment.'),PageBreak()]
    selected_groups=G[(G.split=='test')&(G.policy==PICK)].sort_values('group')
    baseline_groups=G[(G.split=='test')&(G.policy==BASE)].sort_values('group')
    group_table=[]
    for label,frame in [('Baseline',baseline_groups),('Selected',selected_groups)]:
        for r in frame.itertuples():
            group_table.append([label,f'Group {r.group}',f'{r.n:,}',pct(r.selection_rate),pct(r.true_positive_rate),pct(r.false_positive_rate),pct(r.precision)])
    story += [heading('4. What changed for the two groups'),
       table(['Policy','Group','N','Approval','TPR','FPR','Precision'],group_table,[65,57,66,77,77,77,W-419]),Spacer(1,12),
       Image(str(PLOTS/'group_error_rates.png'),width=W,height=W*3.7/10.5),
       small('Figure 3. Approval among applicants who would repay (left) and who would default (right). A lower TPR gap is desirable, but a high FPR still matters to affordability and portfolio risk.'),
       para(f'For the selected policy, the signed TPR difference (group 0 minus group 1) is {pp(ci.loc[PICK,"signed_tpr_group0_minus_group1"])}. '
            f'An approximate 95% binomial interval is [{pp(ci.loc[PICK,"ci95_low"])}, {pp(ci.loc[PICK,"ci95_high"])}]. '
            'This interval is conditional on the fitted policy and this synthetic population. It does not include retraining uncertainty, drift or uncertainty about the data-generating assumptions.'),
       para(f'The selected demographic parity gap is {pp(C.demographic_parity_gap)} and the DP ratio is {C.demographic_parity_ratio:.3f}. '
            'Neither is a lending compliance pass/fail test. Different outcome base rates mean equal opportunity and equal overall approval rates need not coincide.'),
       para('The threshold policy needs group membership when making a decision, while the reweighed model uses it only during training. That difference affects how each method could be governed. Even with accurate group data, a real deployment would need appropriate review of whether and how those attributes may be used.'),PageBreak()]
    log=(RESULTS/'run.log').read_text(errors='replace') if (RESULTS/'run.log').exists() else ''
    log_lines=[line for line in log.splitlines() if line.startswith(('RUN ','TRAIN ','CALIBRATION ','SELECTED ','VERIFIED '))]
    story += [heading('5. Scale, checks and reproduction'),
       para(f'The complete audit ran on {M["rows"]:,} rows with Spark {M["spark_version"]}, {M["master"]}, and {M["partitions"]} partitions. '
            f'The audit stage took {M["elapsed_seconds"]:.1f} seconds ({M["elapsed_seconds"]/60:.1f} minutes). '
            'This is a local execution measurement, not a claim about a production cluster or linear speedup.'),
       table(['Stage','Where it runs / what is collected'],[
       ['Generate and split','Spark SQL hash expressions; all 2,000,000 rows remain distributed'],
       ['Train / reweigh','Spark ML logistic regression; four group-label counts fit weights'],
       ['Calibrate','At most 100,000 validation score rows collected for Fairlearn'],
       ['Evaluate','Spark exports only policy-by-split-by-group confusion counts'],
       ['Cross-check','Fairlearn MetricFrame uses 8 weighted cells per policy/split'],
       ['Document','Small metrics CSVs produce all tables and plots']], [105,W-105]),Spacer(1,12),
       para(f'The maximum Spark/Fairlearn metric difference was {M["max_fairlearn_metric_error"]:.3g}; the maximum exported threshold-probability error was {M["max_export_probability_error"]:.3g}. '
            'Both checks assert an error below 1e-10. Weighted cells reproduce the reported confusion-based metrics exactly; this shortcut cannot substitute for raw scores when checking ranking, calibration or individual fairness.'),
       heading('Recorded run evidence'),small('The following lines are taken from the actual saved run log. They are text evidence, not a screenshot of a different environment.'),
       *[para(escape(line),'CodeCustom') for line in log_lines],
       heading('Reproduce or demonstrate'),
       para('Open the included notebook in Colab and run the cells in order, or follow README.md for a local Java 17 / Python setup. '
            'A 20,000-row smoke run checks the full path quickly; the submitted findings come from the two-million-row run. '
            'No API key is required. The CSVs and model artifacts remain available even when a full rerun is inconvenient.'),
       small('The package includes pinned dependencies, model coefficients, Spark model directories, threshold rules, group confusion counts, the calibration sample, a 1,000-row input example, run metadata and numerical checks. Input generation is reproducible, so the full synthetic table is not bundled.'),PageBreak()]
    refs=[
       '[1] Fairlearn 0.13. ThresholdOptimizer API and postprocessing guide. https://fairlearn.org/v0.13/api_reference/generated/fairlearn.postprocessing.ThresholdOptimizer.html',
       '[2] Hardt, Price and Srebro (2016). Equality of Opportunity in Supervised Learning. https://arxiv.org/abs/1610.02413',
       '[3] Kamiran and Calders (2012). Data preprocessing techniques for classification without discrimination. https://doi.org/10.1007/s10115-011-0463-8',
       '[4] Apache Spark 3.5.3. ML classification and regression; logistic regression. https://spark.apache.org/docs/3.5.3/ml-classification-regression.html#logistic-regression',
       '[5] Mitchell et al. (2019). Model Cards for Model Reporting. https://arxiv.org/abs/1810.03993',
       '[6] Fairlearn. Common fairness metrics and their limitations. https://fairlearn.org/v0.13/user_guide/assessment/common_fairness_metrics.html']
    story += [heading('6. Which metric suits lending?'),
       para('For this case, I would use equal opportunity as the primary applicant-facing metric. With repayment as the positive label, it asks a practical question: among applicants who would repay, how often does each group receive approval? '
            'It directly addresses the harm of denying a viable borrower. Equalized odds is a useful stricter companion because it also checks unequal approval among applicants who would default [2,6].'),
       para('I would keep false positive rate and default among approved applicants as separate risk checks. The selected policy was not optimized for loan size, interest, loss given default or affordability because the brief supplied none of these. '
            'Accuracy gives the two mistakes equal weight; a lender would need an explicit cost and affordability model before setting an operating point.'),
       para('Demographic parity answers a different question: whether groups receive approvals at similar overall rates. It is useful for spotting allocation differences, but it ignores repayment outcomes. '
            'I would not use a single DP ratio, or an imported numerical cutoff, as proof that a lending system is fair.'),
       heading('Limits of the conclusion'),
       para('The data has two fictional groups, complete labels and one controlled form of measurement bias. It does not cover intersectional groups, missing demographics, unequal observation windows, outcome censoring, temporal drift or the effect of an actual loan offer. '
            'Real lenders observe repayment mainly for approved borrowers, so naive error-rate estimates can be affected by selection bias. Simply treating rejected applicants as defaults would invalidate the audit.'),
       para('I would take forward the validation-selected policy for further testing, rather than deploy it. The next checks would use representative time-based holdouts, more demographic slices, repeated training seeds, score calibration and realistic lending costs. '
            'The model card records these boundaries so that a numerical fairness improvement is not mistaken for a complete lending approval.'),
       heading('References'),*[small(escape(r)) for r in refs]]
    build_pdf(OUT/'Credit_Scoring_Fairness_Audit_Report.pdf',story,'Credit Scoring Fairness Audit')


def model_card():
    spec=next(p for p in P if p['policy']==PICK)
    q=G[(G.split=='test')&(G.policy==PICK)].sort_values('group')
    settings=json.dumps(spec,indent=2)
    md=f'''# Model card: synthetic lending repayment model

**Developer:** Kundhan (PES1UG24AM327)  
**Version/date:** 1.0 / 28 October 2026  
**Status:** Academic experiment; not approved for real lending.  
**Selected decision policy:** `{PICK}`  
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

{M['rows']:,} synthetic applications with seed {M['seed']}. Train:
{M['splits']['train']:,}; validation: {M['splits']['validation']:,}; test:
{M['splits']['test']:,}. Hash splits are disjoint. Group 1 prevalence is about
45%. The generator reduces measured history by 0.9 for group 1 but uses true
history for repayment labels. Groups also have different input distributions.
Labels are available for all applications, unlike many real lending datasets.

Three logistic models use L2 regularization 0.01, at most 40 iterations and
weight strengths 0, 0.5 and 1.0. ThresholdOptimizer fits on a bounded validation
sample of {M['threshold_calibration_n']:,} scores. {M['policy_count']} policies
are evaluated. Selection maximizes validation accuracy with TPR gap <=0.02 and
approval between 0.2 and 0.9, with a documented minimum-gap fallback.
Fallback used: {M['selection_fallback_used']}. Test results never enter selection.

## Test results

| Metric | Baseline | Selected |
|---|---:|---:|
| Accuracy | {pct(B.accuracy)} | {pct(C.accuracy)} |
| Balanced accuracy | {pct(B.balanced_accuracy)} | {pct(C.balanced_accuracy)} |
| Equal opportunity gap | {pp(B.equal_opportunity_gap)} | {pp(C.equal_opportunity_gap)} |
| Equalized odds gap | {pp(B.equalized_odds_gap)} | {pp(C.equalized_odds_gap)} |
| Demographic parity gap | {pp(B.demographic_parity_gap)} | {pp(C.demographic_parity_gap)} |
| Demographic parity ratio | {B.demographic_parity_ratio:.3f} | {C.demographic_parity_ratio:.3f} |
| Approval rate | {pct(B.selection_rate)} | {pct(C.selection_rate)} |
| False positive rate | {pct(B.false_positive_rate)} | {pct(C.false_positive_rate)} |
| Default among approved | {pct(B.approved_default_rate)} | {pct(C.approved_default_rate)} |

Equal opportunity is the main applicant-facing metric in this case. A smaller
gap means more similar approval rates among would-repay applicants. It does
not establish fairness across every error type or prove legal compliance.

## Selected policy configuration

```json
{settings}
```

Fairlearn postprocessing can randomize between thresholds. The supplied
pipeline uses a stable ID hash (seed salt 900) to realize those probabilities
in Spark. Re-evaluated outcomes can differ from calibration constraints.
Group membership is required at inference if the chosen policy is a Fairlearn
threshold policy. The actual chosen method is `{spec['method']}`.

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

Spark {M['spark_version']}; Fairlearn {M['packages']['fairlearn']}; Python
{M['python']}. Full audit elapsed {M['elapsed_seconds']:.1f} seconds on
{M['master']} with {M['partitions']} partitions. Maximum Spark/Fairlearn
metric error: {M['max_fairlearn_metric_error']:.3g}. Export probability error:
{M['max_export_probability_error']:.3g}. See requirements.txt, audit.py,
results/run_manifest.json, results/model_coefficients.json and
results/policies.json. Contact/owner for this academic artifact: Kundhan.
Model card structure follows Mitchell et al. (2019),
https://arxiv.org/abs/1810.03993.
'''
    (ROOT/'MODEL_CARD.md').write_text(md)
    story=[small('MODEL CARD / VERSION 1.0'),para('Synthetic lending<br/>repayment model','TitleCustom'),
       para('Kundhan | PES1UG24AM327 | 28 October 2026','SubCustom'),
       para(f'<b>Policy:</b> {PICK}<br/><b>Status:</b> Academic experiment. No production release recommended.'),
       heading('Intended use'),para('Demonstrate fairness assessment at scale for a Big Data case study. Intended users are the student, evaluator and technical reviewers. The model is unsuitable for live underwriting, pricing, credit limits or eligibility decisions.'),
       heading('Model and interface'),para('Spark ML binary logistic regression uses income_z, debt_z, history_measured and employment_z. The inputs are synthetic standardized descriptors; they are not a production bureau schema. The positive label is would repay, and decision 1 means approve. The probability-like repayment score has not received a calibration audit.'),
       para('Group membership is excluded from the logistic feature vector. Reweighing uses it at training time. Fairlearn postprocessing also uses it at decision time. Missing features, unknown groups and nonnumeric production data are outside the validated input range.'),
       heading('Data and training'),para(f'{M["rows"]:,} synthetic applications; seed {M["seed"]}. Train/validation/test sizes are {M["splits"]["train"]:,} / {M["splits"]["validation"]:,} / {M["splits"]["test"]:,}. '
       'The generator includes complete outcome labels, group-dependent feature distributions and a 0.9-unit under-measurement of group 1 credit history. It uses no Home Credit records or actual customers.'),
       para(f'L2 logistic regularization: 0.01; maximum iterations: 40. Reweighing strengths: 0, 0.5, 1. ThresholdOptimizer uses {M["threshold_calibration_n"]:,} validation scores. '
            f'{M["policy_count"]} decision policies are compared. All policy selection uses validation data only.'),
       heading('Decision objective'),para('Maximize validation accuracy with a TPR gap of at most 2 percentage points and approval between 20% and 90%. If infeasible, minimize the validation gap within those approval bounds. '
            f'Fallback used: {M["selection_fallback_used"]}. These are academic operating choices and have no claimed regulatory status.'),PageBreak()]
    rows=[[name,formatting(B[key]),formatting(C[key])] for name,key,formatting in [
       ('Accuracy','accuracy',pct),('Balanced accuracy','balanced_accuracy',pct),('Equal opportunity gap','equal_opportunity_gap',pp),
       ('Equalized odds gap','equalized_odds_gap',pp),('Demographic parity gap','demographic_parity_gap',pp),
       ('Approval rate','selection_rate',pct),('False positive rate','false_positive_rate',pct),('Default among approved','approved_default_rate',pct)]]
    story += [heading('Measured test performance'),table(['Metric','Baseline','Selected'],rows,[W-180,90,90]),Spacer(1,12),
       table(['Selected subgroup','N','TPR','FPR','Approval'],[[f'Group {r.group}',f'{r.n:,}',pct(r.true_positive_rate),pct(r.false_positive_rate),pct(r.selection_rate)] for r in q.itertuples()],[110,90,95,95,W-390]),Spacer(1,12),
       para('Equal opportunity measures approval differences among applicants who would repay. Equalized odds adds false-positive disparity. Both are reported as absolute gaps in percentage points; smaller is better. The default rate among approved applicants is a separate portfolio-risk measure.'),
       heading('Selected decision rule'),
       para(f'The selected method is <b>{spec["method"]}</b>. Full settings are stored in results/policies.json and MODEL_CARD.md. '
       'Fairlearn threshold rules can mix two thresholds per group. The Spark implementation realizes the probabilities using a stable per-application ID hash, with seed salt 900. Calibration and held-out realized rates may differ.'),
       heading('Verification'),para(f'Spark aggregates the complete test split. Fairlearn independently checks confusion-based group and overall metrics using weighted confusion cells. Maximum metric error: {M["max_fairlearn_metric_error"]:.3g}. '
       f'Maximum threshold-export probability error: {M["max_export_probability_error"]:.3g}. These checks validate the implementation, not the representativeness of the synthetic population.'),
       small('All figures refer to this fixed model, population and randomized-decision seed. They do not estimate uncertainty from model retraining or changes to the population.'),PageBreak()]
    story += [heading('Limitations and risks'),para('This experiment has two fictional groups, one seed and one static synthetic population. It does not establish real-world accuracy, affordability or a reduction in actual discrimination. It does not evaluate intersectional groups, individual fairness, ranking quality, score calibration or temporal drift.'),
       para('All synthetic applicants have labels. In real lending, repayment is usually observed only after approval and over a fixed outcome window. Rejected applicants must not automatically be assigned a default label. Missing or selectively observed outcomes can invalidate an apparently precise fairness estimate.'),
       para('A lower TPR gap can coexist with unequal false positive rates or a larger default rate among approved applicants. This run has no loan amounts, pricing, recovery assumptions or quantified error costs. Its selected policy therefore is not a validated economic lending policy.'),
       para('Group-dependent thresholds require protected-attribute data at decision time. Whether such use is appropriate needs review in the actual setting. Randomization also needs an accountable decision procedure, stable logging and an appeal process. A small metric gap is not evidence of legal compliance.'),
       heading('Conditions before any real pilot'),para('Use representative data and a time-based holdout. Define repayment outcomes, maturity windows, loan costs and affordability constraints. Check calibration and errors across relevant groups and intersections. Repeat fitting across seeds and periods. Review protected-data handling and decision rules with appropriate domain reviewers.'),
       heading('Monitoring if a future model is deployed'),para('Track approval rates, outcome maturity, subgroup sample sizes, TPR/FPR, default among approved, calibration and drift. Record model and policy versions, score inputs and randomized-decision seeds. Set alert thresholds and review ownership before rollout. Offer human review and a way to challenge adverse decisions.'),
       heading('Reproduction and ownership'),para(f'Spark {M["spark_version"]}, Fairlearn {M["packages"]["fairlearn"]}, Python {M["python"]}. The audit took {M["elapsed_seconds"]:.1f} seconds on {M["master"]} with {M["partitions"]} partitions. '
       'The package includes source, pinned dependencies, trained models, policy settings, count tables and the run manifest. Academic artifact owner: Kundhan.'),
       small('Documentation structure: Mitchell et al. (2019), Model Cards for Model Reporting. https://arxiv.org/abs/1810.03993')]
    build_pdf(OUT/'Model_Card.pdf',story,'Model Card - Synthetic Lending Repayment Model')


def notebook():
    cells=[]
    def md(s):cells.append({'cell_type':'markdown','metadata':{},'source':s.splitlines(True)})
    def code(s):cells.append({'cell_type':'code','execution_count':None,'metadata':{},'outputs':[],'source':s.splitlines(True)})
    md('# Credit Scoring Fairness Audit\nKundhan · PES1UG24AM327\n\nTwo million **synthetic** applications; Spark training and aggregation, training reweighing and Fairlearn threshold optimization. No API keys needed. Run in Google Colab with a CPU runtime. Existing measured outputs are supplied separately; this notebook has no fabricated cell outputs.')
    code('%pip -q install pyspark==3.5.3 fairlearn==0.13.0 scikit-learn==1.7.2 numpy==2.2.6 pandas==2.3.3 scipy==1.16.3 matplotlib==3.10.7 reportlab==4.4.4 pypdf==6.1.3 pytest==8.4.2\n')
    md('If Colab asks to restart after installing numerical packages, restart the session and continue below. Java 17 and Python 3.11/3.12 are supported. If using a different Python version, use the local setup in README.md.')
    code('import os, sys\nos.environ["PYSPARK_PYTHON"] = sys.executable\nos.environ["SPARK_LOCAL_IP"] = "127.0.0.1"\n!java -version\n')
    for filename in ['fairness_core.py','audit.py','make_deliverables.py','requirements.txt','README.md','tests/test_core.py']:
        if '/' in filename:code('from pathlib import Path\nPath("tests").mkdir(exist_ok=True)\n')
        code('%%writefile '+filename+'\n'+(ROOT/filename).read_text())
    md('## Check the numerical conventions\nThese tests use a hand-calculated confusion matrix, reweighing identities and threshold boundary cases.')
    code('!python -m unittest discover -s tests -v\n')
    md('## Optional smoke run\nThis checks the full Spark/Fairlearn path quickly. It does not replace the complete run below.')
    code('!python audit.py --rows 20000 --out smoke\n')
    md('## Complete audit\nThe log captures the actual execution. Calibration is bounded; final metrics use the full held-out test split. Policy selection uses validation only.')
    code('import subprocess\nfrom pathlib import Path\nPath("results").mkdir(exist_ok=True)\nwith open("results/run.log", "w") as log:\n    run = subprocess.run([sys.executable,"audit.py","--rows","2000000","--out","results"], stdout=log, stderr=subprocess.STDOUT)\nprint(Path("results/run.log").read_text()[-10000:])\nrun.check_returncode()\n')
    code('import pandas as pd, json\nmanifest=json.loads(Path("results/run_manifest.json").read_text())\nmetrics=pd.read_csv("results/metrics.csv")\ndisplay(metrics[(metrics.split=="test") & (metrics.policy.isin(list(manifest["representatives"].values())+[manifest["selected_policy"]]))])\n')
    md('## Build plots, report, model card and submission ZIP\nEvery document value is populated from this run. The charts display held-out test results; the chosen policy remains the validation winner.')
    code('!python make_deliverables.py\nfrom IPython.display import Image, display\ndisplay(Image("plots/accuracy_fairness_tradeoff.png"))\ndisplay(Image("plots/group_error_rates.png"))\n')
    code('from google.colab import files\nfiles.download("outputs/Credit_Scoring_Fairness_Audit_Submission.zip")\n')
    nb={'cells':cells,'metadata':{'kernelspec':{'display_name':'Python 3','language':'python','name':'python3'},'language_info':{'name':'python','version':'3.11'},'colab':{'name':'Credit_Scoring_Fairness_Audit.ipynb'}},'nbformat':4,'nbformat_minor':5}
    for i,c in enumerate(cells):c['id']=f'cell-{i:03d}'
    (ROOT/'Credit_Scoring_Fairness_Audit.ipynb').write_text(json.dumps(nb,indent=1,ensure_ascii=False))


def package():
    included=['audit.py','fairness_core.py','make_deliverables.py','requirements.txt','README.md','MODEL_CARD.md','Credit_Scoring_Fairness_Audit.ipynb']
    if (ROOT/'pytest.ini').exists():included.append('pytest.ini')
    target=OUT/'Credit_Scoring_Fairness_Audit_Submission.zip'
    with zipfile.ZipFile(target,'w',zipfile.ZIP_DEFLATED) as z:
        for name in included:z.write(ROOT/name,'Credit_Scoring_Fairness_Audit/'+name)
        for folder in ['results','plots','tests']:
            for path in sorted((ROOT/folder).rglob('*')):
                if path.is_file() and '__pycache__' not in path.parts and not path.name.startswith('.'):
                    z.write(path,'Credit_Scoring_Fairness_Audit/'+str(path.relative_to(ROOT)))
        for name in ['Credit_Scoring_Fairness_Audit_Report.pdf','Model_Card.pdf']:
            z.write(OUT/name,'Credit_Scoring_Fairness_Audit/'+name)
    print(f'Created {target.name} ({target.stat().st_size:,} bytes)')


if __name__=='__main__':
    charts();ci=uncertainty();report(ci);model_card();notebook();package()
    print('Built measured charts, report, model card, notebook and complete submission ZIP.')

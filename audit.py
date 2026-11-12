"""Reproducible Spark/Fairlearn lending fairness audit. See README.md."""
import argparse
import importlib.metadata as metadata
import json
import math
import os
from pathlib import Path
import platform
import time
from datetime import datetime, timezone
import numpy as np
import pandas as pd
from pyspark.sql import SparkSession, functions as F
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.classification import LogisticRegression
from pyspark.ml.functions import vector_to_array
from fairlearn.postprocessing import ThresholdOptimizer
from fairlearn.metrics import MetricFrame, selection_rate, true_positive_rate, false_positive_rate
from sklearn.metrics import accuracy_score
from fairness_core import ScoreAdapter, count_metrics, reweighing_table, threshold_probability

FEATURES=['income_z','debt_z','history_measured','employment_z']
SEED=327


def uniform(salt):
    # Hashing by ID makes generation invariant to partition count and row order.
    return (F.pmod(F.xxhash64('id', F.lit(SEED), F.lit(salt)), F.lit(1000000000))+.5)/1e9


def normal(salt):
    return F.sqrt(-2*F.log(uniform(salt)))*F.cos(2*math.pi*uniform(salt+1))


def synthetic(spark,n,partitions):
    df=spark.range(n,numPartitions=partitions).withColumn('group',(uniform(1)<.45).cast('int'))
    df=(df.withColumn('income_z',normal(10)-.35*F.col('group'))
        .withColumn('debt_z',normal(20)+.25*F.col('group'))
        .withColumn('history_true',normal(30))
        .withColumn('employment_z',normal(40)-.15*F.col('group')))
    z=1.1+1.0*F.col('income_z')-1.2*F.col('debt_z')+.9*F.col('history_true')+.45*F.col('employment_z')
    df=(df.withColumn('repay_probability',1/(1+F.exp(-z)))
        .withColumn('label',(uniform(50)<F.col('repay_probability')).cast('double'))
        .withColumn('history_measured',F.col('history_true')-.9*F.col('group'))
        .withColumn('bucket',F.pmod(F.xxhash64('id',F.lit('split327')),F.lit(10)))
        .withColumn('split',F.when(F.col('bucket')<6,'train').when(F.col('bucket')<8,'validation').otherwise('test')))
    return VectorAssembler(inputCols=FEATURES,outputCol='features').transform(df).select('id','group','label','split',*FEATURES,'features')


def export_rule(opt):
    result={}
    for group,rule in opt.interpolated_thresholder_.interpolation_dict.items():
        result[str(int(group))]={
            'p0':float(rule.p0),'p1':float(rule.p1),
            'operation0':{'operator':rule.operation0.operator,'threshold':float(rule.operation0.threshold)},
            'operation1':{'operator':rule.operation1.operator,'threshold':float(rule.operation1.threshold)}}
    return result


def spark_probability(rule,score):
    def op(spec):
        value=F.col(score)
        return (value>F.lit(spec['threshold'])).cast('double') if spec['operator']=='>' else (value<F.lit(spec['threshold'])).cast('double')
    return rule['p0']*op(rule['operation0'])+rule['p1']*op(rule['operation1'])


def aggregate_policies(scored,policies):
    entries=[]
    for p in policies:
        if p['method']=='threshold_optimizer':
            pr=F.when(F.col('group')==0,spark_probability(p['rules']['0'],'score_0')).otherwise(spark_probability(p['rules']['1'],'score_0'))
            prediction=(uniform(900)<pr).cast('int')
        else:
            prediction=(F.col(p['score'])>p['threshold']).cast('int')
        entries.append(F.struct(F.lit(p['policy']).alias('policy'),prediction.alias('prediction')))
    long=scored.select('split','group','label',F.explode(F.array(*entries)).alias('p')).select('split','group','label','p.*')
    return long.groupBy('split','policy','group').agg(*[
        F.sum(F.when((F.col('label')==y)&(F.col('prediction')==p),1).otherwise(0)).alias(name)
        for y,p,name in [(0,0,'tn'),(0,1,'fp'),(1,0,'fn'),(1,1,'tp')]]).toPandas()


def verify_fairlearn(counts,overall,groups):
    # 8 weighted representatives are exactly equivalent for confusion-derived metrics.
    y=[];pred=[];sensitive=[];weights=[]
    for r in counts.to_dict('records'):
        for yy,pp,key in [(0,0,'tn'),(0,1,'fp'),(1,0,'fn'),(1,1,'tp')]:
            y.append(yy);pred.append(pp);sensitive.append(int(r['group']));weights.append(int(r[key]))
    funcs={'accuracy':accuracy_score,'selection_rate':selection_rate,
           'true_positive_rate':true_positive_rate,'false_positive_rate':false_positive_rate}
    mf=MetricFrame(metrics=funcs,y_true=y,y_pred=pred,sensitive_features=sensitive,
                   sample_params={k:{'sample_weight':weights} for k in funcs})
    errors=[abs(float(mf.overall[k])-overall[k]) for k in funcs]
    for g in groups:
        errors.extend(abs(float(mf.by_group.loc[g['group'],k])-g[k]) for k in funcs)
    dif=mf.difference(method='between_groups')
    errors.extend([abs(float(dif.selection_rate)-overall['demographic_parity_gap']),
                   abs(float(dif.true_positive_rate)-overall['equal_opportunity_gap']),
                   abs(max(float(dif.true_positive_rate),float(dif.false_positive_rate))-overall['equalized_odds_gap'])])
    error=max(errors)
    if error>1e-10: raise AssertionError(f'Spark/Fairlearn disagreement {error}')
    return error


def run(args):
    start=time.time(); out=Path(args.out);out.mkdir(parents=True,exist_ok=True)
    os.environ.setdefault('SPARK_LOCAL_IP','127.0.0.1')
    spark=(SparkSession.builder.master(args.master).appName('CreditScoringFairnessAudit')
           .config('spark.sql.shuffle.partitions',str(args.partitions))
           .config('spark.driver.bindAddress','127.0.0.1')
           .config('spark.driver.memory','3g').config('spark.ui.enabled','false').getOrCreate())
    spark.sparkContext.setLogLevel('ERROR')
    print(f'RUN rows={args.rows:,} Spark={spark.version} seed={SEED}',flush=True)
    df=synthetic(spark,args.rows,args.partitions).cache()
    profile=df.groupBy('split','group','label').count().toPandas()
    assert int(profile['count'].sum())==args.rows
    profile.to_csv(out/'data_profile.csv',index=False)
    df.select('id','group','label','split',*FEATURES).limit(1000).toPandas().to_csv(out/'sample_applications.csv',index=False)
    print(profile.groupby('split')['count'].sum().to_string(),flush=True)
    train=df.filter(F.col('split')=='train')
    w=reweighing_table(profile[profile.split=='train'][['group','label','count']])
    w.to_csv(out/'reweighing_weights.csv',index=False)
    weight=F.lit(1.)
    for row in w.to_dict('records'):
        weight=F.when((F.col('group')==int(row['group']))&(F.col('label')==float(row['label'])),F.lit(float(row['weight']))).otherwise(weight)
    train=train.withColumn('independence_weight',weight)
    scored=df.filter(F.col('split')!='train')
    policies=[]; model_specs=[]
    for i,alpha in enumerate([0.,.5,1.]):
        tick=time.time()
        fitted=LogisticRegression(maxIter=40,regParam=.01,tol=1e-6,weightCol='weight').fit(train.withColumn('weight',(1-alpha)+alpha*F.col('independence_weight')))
        fitted.write().overwrite().save(str((out/'models'/f'lr_{i}').resolve()))
        score=f'score_{i}'
        scored=fitted.transform(scored).withColumn(score,vector_to_array('probability')[1]).drop('probability','rawPrediction','prediction')
        model_specs.append({'model':f'lr_{i}','alpha':alpha,'features':FEATURES,'intercept':float(fitted.intercept),'coefficients':fitted.coefficients.toArray().tolist(),'seconds':time.time()-tick})
        for threshold in np.arange(.30,.851,.05):
            policies.append({'policy':f'lr{i}_t{threshold:.2f}','method':'baseline' if i==0 else 'reweighing','alpha':alpha,'score':score,'threshold':round(float(threshold),2)})
        print(f'TRAIN alpha={alpha} seconds={time.time()-tick:.1f}',flush=True)
    scored=scored.select('id','group','label','split','score_0','score_1','score_2').cache()
    scored.count()
    # Bound the only application-level collection. Sampling is independent of label/group.
    val=scored.filter((F.col('split')=='validation')&(uniform(700)<.24)).select('id','group','label','score_0').limit(100000).toPandas()
    adapter=ScoreAdapter()
    export_errors=[]
    for tol in [0.,.01,.025,.05,.10]:
        opt=ThresholdOptimizer(estimator=adapter,constraints='true_positive_rate_parity',objective='accuracy_score',
                               predict_method='predict_proba',prefit=True,grid_size=1000,tol=tol)
        opt.fit(val[['score_0']],val.label,sensitive_features=val.group)
        rules=export_rule(opt)
        # Check exported probabilities against Fairlearn itself on the calibration rows.
        reference=opt.interpolated_thresholder_._pmf_predict(val[['score_0']],sensitive_features=val.group)[:,1]
        exported=np.zeros(len(val))
        for g in [0,1]:
            mask=val.group.to_numpy()==g
            exported[mask]=threshold_probability(val.score_0.to_numpy()[mask],rules[str(g)])
        error=float(np.max(np.abs(reference-exported)));export_errors.append(error)
        assert error<1e-10
        policies.append({'policy':f'to_tol{tol:.3f}','method':'threshold_optimizer','tolerance':tol,'rules':rules})
    val.to_csv(out/'threshold_calibration_sample.csv',index=False)
    print(f'CALIBRATION rows={len(val):,}; policy count={len(policies)}; export max error={max(export_errors)}',flush=True)
    counts=aggregate_policies(scored,policies)
    counts.to_csv(out/'confusion_counts.csv',index=False)
    summaries=[]; group_rows=[]; checks=[]
    for (split,policy),subset in counts.groupby(['split','policy']):
        overall,groups=count_metrics(subset)
        checks.append(verify_fairlearn(subset,overall,groups))
        spec=next(p for p in policies if p['policy']==policy)
        summaries.append(dict(split=split,policy=policy,method=spec['method'],**overall))
        group_rows.extend(dict(split=split,policy=policy,**g) for g in groups)
    summary=pd.DataFrame(summaries);summary.to_csv(out/'metrics.csv',index=False)
    pd.DataFrame(group_rows).to_csv(out/'group_metrics.csv',index=False)
    validation=summary[summary.split=='validation']
    # Predeclared academic criterion: maximize accuracy among nontrivial policies
    # with <=2 percentage point TPR gap and approval rate in [0.2,0.9].
    eligible=validation[(validation.equal_opportunity_gap<=.02)&validation.selection_rate.between(.2,.9)]
    fallback=eligible.empty
    if fallback:
        selected=validation[validation.selection_rate.between(.2,.9)].sort_values(['equal_opportunity_gap','accuracy'],ascending=[True,False]).iloc[0]
    else:
        selected=eligible.sort_values(['accuracy','equal_opportunity_gap','policy'],ascending=[False,True,True]).iloc[0]
    representatives={method:validation[validation.method==method].sort_values(['accuracy','policy'],ascending=[False,True]).iloc[0].policy for method in ['baseline','reweighing','threshold_optimizer']}
    manifest={'dataset':'synthetic; not Home Credit; no real applicants','rows':args.rows,'seed':SEED,
              'timestamp_utc':datetime.now(timezone.utc).isoformat(),'spark_version':spark.version,
              'master':args.master,'partitions':args.partitions,'python':platform.python_version(),'platform':platform.platform(),
              'packages':{k:metadata.version(k) for k in ['pyspark','fairlearn','numpy','pandas','scikit-learn','scipy']},
              'splits':{k:int(v) for k,v in profile.groupby('split')['count'].sum().items()},
              'threshold_calibration_n':len(val),'calibration_cap':100000,
              'selected_policy':selected.policy,'selection_rule':'validation accuracy maximum with TPR gap <=0.02 and approval rate 0.2..0.9; if infeasible minimize TPR gap within approval bounds',
              'selection_fallback_used':bool(fallback),'representatives':representatives,
              'max_fairlearn_metric_error':max(checks),'max_export_probability_error':max(export_errors),
              'policy_count':len(policies),'elapsed_seconds':time.time()-start}
    for name,data in [('run_manifest.json',manifest),('policies.json',policies),('model_coefficients.json',model_specs)]:
        (out/name).write_text(json.dumps(data,indent=2,allow_nan=False))
    print('SELECTED '+selected.policy,flush=True)
    print(summary[(summary.split=='test')&summary.policy.isin(set(representatives.values())|{selected.policy})][['policy','accuracy','equal_opportunity_gap','equalized_odds_gap','false_positive_rate']].to_string(index=False),flush=True)
    print(f'VERIFIED Fairlearn max error={max(checks)}; elapsed={manifest["elapsed_seconds"]:.1f}s',flush=True)
    spark.stop()


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--rows',type=int,default=2000000)
    parser.add_argument('--out',default='results')
    parser.add_argument('--master',default='local[4]')
    parser.add_argument('--partitions',type=int,default=8)
    args=parser.parse_args()
    if args.rows<10000: parser.error('Use at least 10,000 rows for stable calibration cells')
    run(args)

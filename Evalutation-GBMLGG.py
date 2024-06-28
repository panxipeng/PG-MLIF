from core.utils_data import getCleanGBMLGG
from core.utils_analysis import *
from tqdm import tqdm



# Glioma Survival Outcome Prediction


### 1. Cox Baselines

trainCox_GBMLGG(model='cox_agegender', penalizer=0)
trainCox_GBMLGG(model='cox_moltype', penalizer=0)
trainCox_GBMLGG(model='cox_grade', penalizer=0)
trainCox_GBMLGG(model='cox_molgrade', penalizer=0)


### 2. Statistical Significance + Ensembling Effects in Pathomic Fusion

# models = ['cox_agegender', 'cox_moltype', 'cox_grade', 'cox_molgrade',
#           'omic_silu2_2splits_15',
#           'path_2splits_15',
#           'pathomic_lmfskipsilu2_2splits_15']
models = ['cox_agegender', 'cox_moltype', 'cox_grade', 'cox_molgrade',
          'omic_silu2_2splits_15',
          'path_2splits_15',
          'pathomic_CTrans']
model_names = ['Cox (Age+Gender)', 'Cox (Subtype)', 'Cox (Grade)', 'Cox (Grade+Subtype)',
               'Genomic SNN',
               'Histology CNN',
               'MLIF:Pathomic.(CNN+SNN)']


ckpt_name = '/home/yons/An/paper/20220318-PathomicFusion/20230529PO-Fusion/checkpoints/TCGA_GBMLGG/surv_15_rnaseq/'
pvalue_surv_multi = [np.array(getPValAggSurv_GBMLGG_Multi(ckpt_name=ckpt_name, model=model, percentile=[33,66])) for model in tqdm(models)]
pvalue_surv_multi = pd.DataFrame(np.array(pvalue_surv_multi))
pvalue_surv_binary = [np.array(getPValAggSurv_GBMLGG_Binary(ckpt_name=ckpt_name, model=model, percentile=[50])) for model in tqdm(models)]
pvalue_surv_binary = pd.DataFrame(np.array(pvalue_surv_binary))
pvalue_surv = pd.concat([pvalue_surv_binary, pvalue_surv_multi],axis=1)
pvalue_surv.index = model_names
pvalue_surv.columns = ['P-Value (<50% vs. >50%)', 'P-Value (<33% vs. 33-66%)', 'P-Value (33-66% vs. >66%)']
#
cv_surv = [np.array(getPredAggSurv_GBMLGG(ckpt_name=ckpt_name, model=model)) for model in tqdm(models)]
cv_surv = pd.DataFrame(np.array(cv_surv))
cv_surv.columns = ['Split %s' % str(k) for k in range(1,16)]
cv_surv.index = model_names
cv_surv['C-Index'] = [CI_pm(cv_surv.loc[model]) for model in model_names]
cv_surv_all = cv_surv[['C-Index']].join(pvalue_surv, how='inner')
#
#
print(cv_surv)
print(cv_surv_all)
#
# ### 3. Plots
#
cluster_p = getHazardHistogramPlot_GBMLGG(model='path_2splits_15', c=[(-1.0, -0.5), (0, 0.75), (1, 1.5)])
cluster_o = getHazardHistogramPlot_GBMLGG(model='omic_silu2_2splits_15', c=[(-1.0, -0.5), (1, 1.25), (1.25, 1.5)])
cluster_pgo = getHazardHistogramPlot_GBMLGG(model='PG-MLIF', c=[(-1.0, -0.5), (1, 1.25), (1.25, 1.5)])
cluster_results = pd.concat([cluster_p, cluster_o, cluster_pgo])
# cluster_results

for model in tqdm(['path_2splits_15', 'omic_silu2_2splits_15', 'PG-MLIF']):
    makeHazardBoxPlot(ckpt_name='./checkpoints/TCGA_GBMLGG/surv_15_rnaseq/', model=model)
    makeHazardSwarmPlot(ckpt_name='./checkpoints/TCGA_GBMLGG/surv_15_rnaseq/', model=model)
    makeKaplanMeierPlot(ckpt_name='./checkpoints/TCGA_GBMLGG/surv_15_rnaseq/', model=model)

for model in tqdm(['path_2splits_15', 'omic_silu2_2splits_15','PG-MLIF']):
    makeKaplanMeierPlot(ckpt_name='./checkpoints/TCGA_GBMLGG/surv_15_rnaseq/', model=model, plot_gt=False)

makeKaplanMeierPlot_Baseline(model='Histomolecular subtype')

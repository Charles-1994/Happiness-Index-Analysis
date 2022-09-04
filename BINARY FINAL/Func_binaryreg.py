import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import time
from scipy.stats._stats import _kendall_dis
import warnings
warnings.filterwarnings("ignore")

def responseprofile(y):
    ydf = pd.DataFrame(y)
    ydf.columns = ['y'] 
    tata=pd.crosstab(index=ydf['y'], columns='count')
    tata[['count']]
    tata.index.name = 'values'
    tata.reset_index(inplace=True)
    the_table=pd.DataFrame(data=tata)
    the_table.columns = ['Values (y)' , 'Total Frequency']
    plt.subplot(212)
    plt.title('Response Profile',fontsize=20,y=1.7)
    plt.axis('off')
    plt.axis('tight')
    test2=plt.table(cellText=the_table.values, loc='center', cellLoc='center', colLabels=the_table.columns, colWidths=[0.4,0.4])
    test2.auto_set_font_size(False)
    test2.set_fontsize(16) 
    test2.scale(2, 1.7)

def convergence(y,logit_res):    
    val_listsetting = list(logit_res.mle_settings.values()) 
    val_listretvals = list(logit_res.mle_retvals.values()) 
    the_table2 = [ ['Optimizer',val_listsetting[0] ] ,   
                      ['Starting parameters', val_listsetting[1]],
                      ['Max. iterations', val_listsetting[2]], 
                      ['Tolerance rate', val_listsetting[9]],
                      ['Req. iterations', val_listretvals[1]],
                      ['Convergence Status', val_listretvals[5]]]
    the_table2=pd.DataFrame(data=the_table2)
    plt.subplot(212)
    #plt.title('Model Convergence Status',fontsize=20,y=2)
    plt.axis('off')
    plt.axis('tight')
    test2=plt.table(cellText=the_table2.values, loc='center', cellLoc='center', colWidths=[0.4,0.4,0.4])
    test2.auto_set_font_size(False)
    test2.set_fontsize(16) 
    test2.scale(2, 1.7)
    
def informationcriteria(y,logit_res):
    the_table3 = [['AIC', np.round(-2*(logit_res.llnull-1),3),np.round(logit_res.aic,3)],
                      ['BIC', np.round(-2*logit_res.llnull+np.log(logit_res.nobs),3),np.round(logit_res.bic,3)], 
                      ['-2LogL', np.round(-2*logit_res.llnull,3),np.round(-2*logit_res.llf,3)]]
    the_table3=pd.DataFrame(data=the_table3)
    the_table3.columns = ['Criterion', 'Intercept Only', 'Intercept and Covariates']
    plt.subplot(311)
    plt.title('Information Criteria',fontsize=20, y=1.1)
    plt.axis('off')
    plt.axis('tight')
    test3=plt.table(cellText=the_table3.values, colLabels=the_table3.columns, loc='center', cellLoc='center', colWidths=[0.4,0.4,0.4])
    test3.auto_set_font_size(False)
    test3.set_fontsize(16) 
    test3.scale(2, 1.7)
    
def globalnull(y,logit_res):
    M = np.identity(len(logit_res.params))
    M = M[1:,:]
    wald_global = logit_res.wald_test(M,scalar=False) 

    globalnull = [['Likelihood Ratio Test',round(logit_res.llr,4),round(logit_res.df_model),round(logit_res.llr_pvalue,5)],
                 ['Wald',round(wald_global.statistic.item(),4), round(logit_res.df_model),round(wald_global.pvalue.item(),5)]]
    globalnull=pd.DataFrame(data=globalnull)
    globalnull.columns = ['Test','Chi-Square','DF','Pr>ChiSq' ]
    plt.subplot(312)
    plt.title('Testing Global Null Hypothesis: BETA=0',fontsize=20)
    plt.axis('off')
    plt.axis('tight')
    test=plt.table(cellText=globalnull.values, colLabels=globalnull.columns, 
                   loc='center', cellLoc='center', colWidths=[0.3,0.2,0.2,0.2])
    test.auto_set_font_size(False)
    test.set_fontsize(18) 
    test.scale(2, 1.7)

def oddsratioestimates(y,logit_res):
    freqy=y.value_counts()
    m=len(freqy)
    params = logit_res.params[1:len(logit_res.params)]
    conf = logit_res.conf_int()
    conf=conf.iloc[1:len(logit_res.params)]
    confodr=np.exp(conf)
    odr=np.exp(params.to_frame())
    odr=odr.rename(columns={0:'Point Estimate'})
    confodr=confodr.rename(columns={0:'5%', 1:'95%'})
    table=pd.concat([odr[['Point Estimate']], confodr[['5%','95%']]], axis=1)
    plt.subplot(313)
    #plt.title('Odds Ratio Estimates',fontsize=20,y=4)
    plt.axis('off')
    plt.axis('tight')
    table.update(table.astype(float))
    table.update(table.applymap('{:,.3f}'.format))
    test=plt.table(cellText=table.values, colLabels=['Point Estimate', 'Lower CI (95% Wald)','Upper CI (95% Wald)'],  
                   rowLabels=table.index, loc='center',cellLoc='center',colWidths=[0.3,0.35,0.35])
    plt.axis('off')
    test.auto_set_font_size(False)
    test.set_fontsize(15) 
    test.scale(2, 1.7)
    print('!!ATTENTION: Odds ratio are only OK with the Logit link function!!')

def associationstat(X, y, logit_res):
    
    Xmat = np.array(X)
    ymat=np.array(y)
    beta_hat=np.array(logit_res.params)
    #print(np.shape(Xmat))
    #print(np.shape(beta_hat))
    score_hat=Xmat.dot(beta_hat)                    
    x=score_hat

    freqy=y.value_counts()
    total=freqy.iloc[0]*freqy.iloc[1] 
    
    x = np.asarray(x).ravel()
    y = np.asarray(y).ravel()

    if x.size != y.size:
        raise ValueError("All inputs must be of the same size, "
                         "found x-size %s and y-size %s" % (x.size, y.size))

    def count_rank_tie(ranks):
        cnt = np.bincount(ranks).astype('int64', copy=False)
        cnt = cnt[cnt > 1]
        return ((cnt * (cnt - 1) // 2).sum(),
            (cnt * (cnt - 1.) * (cnt - 2)).sum(),
            (cnt * (cnt - 1.) * (2*cnt + 5)).sum())

    size = x.size
    perm = np.argsort(y)  # sort on y and convert y to dense ranks
    x, y = x[perm], y[perm]
    y = np.r_[True, y[1:] != y[:-1]].cumsum(dtype=np.intp)

    # stable sort on x and convert x to dense ranks
    perm = np.argsort(x, kind='mergesort')
    x, y = x[perm], y[perm]
    x = np.r_[True, x[1:] != x[:-1]].cumsum(dtype=np.intp)

    nd = _kendall_dis(x, y)  # discordant pairs

    obs = np.r_[True, (x[1:] != x[:-1]) | (y[1:] != y[:-1]), True]
    cnt = np.diff(np.where(obs)[0]).astype('int64', copy=False)

    ntie = (cnt * (cnt - 1) // 2).sum()  # joint ties
    xtie, x0, x1 = count_rank_tie(x)     # ties in x, stats
    ytie, y0, y1 = count_rank_tie(y)     # ties in y, stats

    tot = (size * (size - 1)) // 2
    
    nc = tot - nd - (xtie - ntie) - (ytie - ntie) - ntie

    #return (nc,nd,ntie,total)
    #nc, nd, ntie, total = AssociationStat(x, y)
    
    the_table = [ ['Percent Concordant',round(nc/total*100,2), 'Somers D',round((nc-nd)/total,3)],   
                      ['Percent Discordant', round(nd/total*100,2), 'Gamma',round((nc-nd)/(nc+nd),3)],
                      ['Percent Tied', round((total-nc-nd)/total*100,2), 'Tau-a', round((nc-nd)/(0.5*y.size*(y.size-1)),3)],
                      ['Pairs', round(total),'c',round((nc+0.5*(total-nc-nd))/total,3)]]
    the_table=pd.DataFrame(data=the_table)
    plt.subplot(221)
    plt.title('Association of Predicted Probabilities and Observed Responses',fontsize=20,y=0.95)
    plt.axis('off')
    plt.axis('tight')
    test=plt.table(cellText=the_table.values, loc='center', cellLoc='center', colWidths=[0.8,0.4,0.4,0.4,0.4])
    test.auto_set_font_size(False)
    test.set_fontsize(18) 
    test.scale(2, 1.7)
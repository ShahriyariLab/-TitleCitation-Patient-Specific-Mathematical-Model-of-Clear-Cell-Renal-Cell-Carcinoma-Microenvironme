#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 24 10:35:44 2022

@author: dilruba
"""

import pandas as pd
import csv
import os
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import seaborn as sns
import scipy as sp
from qspmodel import *
import numpy as np

# Checking or creating necessary output folders
if not os.path.exists('Data/'):
    os.makedirs('Data/Dynamic/')
    os.makedirs('Data/GlobalSensitivity/')
else:
    if not os.path.exists('Data/Dynamic/'):
        os.makedirs('Data/Dynamic/')
    if not os.path.exists('Data/GlobalSensitivity/'):
        os.makedirs('Data/GlobalSensitivity/')

# some global parameters
lmod=[0, 1, 2, 4, 5, 6, 7, 8, 9]  #indices of immune cells variables in cell data (Excluding Naive-cells from the microenvironment)
clusters=4 #number of clusters

T=5000
t=np.linspace(0, T, 30001)

nvar=Renal_QSP_Functions().nvar # number of variables
nparam=Renal_QSP_Functions().nparam # number of parameters
################################################################################
###########################Reading data#########################################
clustercells = pd.read_csv('input/SS_dim_6_3.csv')
clustercells = clustercells.to_numpy()

meanvals = pd.read_csv('input/meanvals_dim_6_3.csv')
meanvals = meanvals.to_numpy()

for cluster in range(clusters):
    QSP_=QSP.from_cell_data(clustercells[cluster])
    params=QSP_.p
    np.savetxt('Cluster'+str(cluster+1)+'params.csv', params, delimiter=",")
    print(params)

################################################################################
###########################Solving ODE##########################################
# reading initial conditions
IC=pd.read_csv('input/IC_ND_6_3.csv')  #non-dimensional IC
IC = IC.to_numpy()
for cluster in range(clusters):
     print('Starting computations for cluster '+str(cluster+1))
     filename='Cluster-'+str(cluster+1)+'-results-'

     QSP_=QSP.from_cell_data(clustercells[cluster])
     params=QSP_.p

     print(' Parameters set. Computing the solution')

     u, _ = QSP_.solve_ode(t, IC[cluster], 'given')
     u = clustercells[cluster]*u

     wr=np.empty((t.size, 16))
     wr[:,0]=t
     wr[:,1:]=u
     c=csv.writer(open('Data/Dynamic/'+filename+'dat.csv',"w"))
     c.writerow(['time']+QSP_.variable_names())
     c.writerows(wr)
     del c
usr_inpt = input('Do you want to plot the dynamics? (yes=1, no=0)')

if int(usr_inpt) == 1:
    dynamic_all = []
    for c in [1,2,3,4]:
        dat = pd.read_csv('Data/Dynamic/Cluster-'+str(c)+'-results-dat.csv')
        dat['Total cells'] = dat[dat.columns[1]]+dat[dat.columns[2]]+dat[dat.columns[3]]+dat[dat.columns[5]]+dat[dat.columns[6]]+0.2*dat[dat.columns[7]]+dat[dat.columns[8]]+dat[dat.columns[9]]+dat[dat.columns[10]]
        dat['Cluster'] = c
        dynamic_all.append(dat)

    dynamic_all_df = pd.concat(dynamic_all, axis=0)

    palette={'Cluster 1':'#3F9B0B', 'Cluster 2':'#FF796C', 'Cluster 3':'#0343DF','Cluster 4':'#000000'}
    dynamic_all_df['Cluster'] = dynamic_all_df['Cluster'].apply(lambda x: 'Cluster '+str(x))
    custom_lines = [Line2D([0], [0], color='#3F9B0B', lw=1.5),
                    Line2D([0], [0], color='#FF796C', lw=1.5),
                    Line2D([0], [0], color='#0343DF', lw=1.5),
                    Line2D([0], [0], color='#000000', lw=1.5)]
    fig, axs = plt.subplots(5, 3, sharey=False, figsize=(11.5,13))
    fig.subplots_adjust(wspace=0.45, hspace=0.6)
    axs = axs.flatten()

    for i, col in enumerate(dynamic_all_df.columns[1:-2]):
        sns.lineplot(data=dynamic_all_df, x='time', y=col, hue='Cluster', palette=palette, ax=axs[i], legend=False)
        axs[i].margins(x=0)
        axs[i].ticklabel_format(axis="y", style="sci", scilimits=(0,0))
        axs[i].set_xlabel("time (days)",fontsize=14)
        axs[i].set_ylabel(str(col),fontsize=14)
    lgd=axs[8].legend(custom_lines, ['Cluster 1', 'Cluster 2', 'Cluster 3', 'Cluster 4'], bbox_to_anchor=(1.1, 0.5), loc='center left',fontsize='small')
    plt.savefig('fig/dynamics5000days.eps', format='eps',dpi=300, bbox_inches='tight') #bbox_inches='tight' makes sure the legends are included in the saved figure
    plt.show()

################################################################################
#######################Sensitivity Analysis#####################################
    
print('Starting steady state global sensitivity analysis')



# Read the parameter perturbation grid
globalgridname='lhs52-5000'
paramscalinggrid = pd.read_csv('input/'+globalgridname+'.csv', header=None).to_numpy().T
lhsnum=paramscalinggrid.shape[0]
# modify from (0,1) range to fit our needs
paramscalinggrid[:,:52]=pow(10,4*paramscalinggrid[:,:52]-2) # scale to values between 0.01 and 100 (10^-2 to 10^2)

# Read the local parameter perturbation grid
# level 1 or 0 corresponds to no local perturbation
gridlevel=2
sensitivity_radius=1 # percentage for local perturbation
if gridlevel>1:
    localgridname='Local-level'+str(gridlevel)
    filename='grid67-level'+str(gridlevel)
    data = pd.read_csv('input/'+filename+'.csv', header=None).to_numpy()
    w=data[:,0]
    x=data[:,1:]
    del data, filename
else:
    localgridname='Local-nogrid'
    w=np.array([1])
    x=[0]

# coefficients for variable sensitivity
lambda0=np.zeros((nvar,2))
lambda0[8,0]=1 # just cancer

import time

start = time.time()
k_filter_all_clusters = []

usr_inpt1 = input("Type in 1 if you have already calculated the global sensitivity. Otherwise type int 0:")

if int(usr_inpt1)!=1:

    import time

    start = time.time()
    k_filter_all_clusters = []

    for cluster in range(0,4):
        print('Starting computations for cluster '+str(cluster+1))

        filename='V67-'+globalgridname+'-'+localgridname+'-cluster-'+str(cluster+1)+'-results-'

        # Calculating all the parameters and weights
        basecore=Renal_QSP_Functions()
        baseparams=op.fsolve((lambda par,frac: basecore.SS_system(par,frac)),
                                np.ones(nparam),args=(clustercells[cluster],))

        paramarray=np.empty((lhsnum,nparam))
        weights=np.empty(lhsnum)
        nonnegs=np.ones(lhsnum, dtype=bool)
        for k in range(lhsnum):
            qspcore=Renal_QSP_Functions(SSrestrictions=paramscalinggrid[k])
            paramarray[k]=op.fsolve((lambda par,frac: qspcore.SS_system(par,frac)),
                                    np.ones(nparam),args=(clustercells[cluster],))
            if (paramarray[k]<0).any(): nonnegs[k]=False
            weights[k]=np.sqrt(((paramarray[k]-baseparams)**2).sum())

        weights/=weights[nonnegs].min()
        weights=np.exp(-weights)
        weights/=weights[nonnegs].sum()

        lambda0[lmod,1]=clustercells[cluster,lmod]/np.sum(clustercells[cluster,lmod]) # all cells except T_N
        dudp=np.zeros((nparam,2))

        k_filter = []

        for k in range(lhsnum):
            if nonnegs[k]:
                for l in range(w.size):
                    QSP_=QSP(paramarray[k]*(1+(sensitivity_radius*1e-2)*x[l]))
                    cancer_sensitivity = np.dot(np.abs(QSP_.Sensitivity()),lambda0)[:,0]
                    # filter out singularities
                    if any(x > 2000 for x in cancer_sensitivity):
                        k_filter.append(k)
                        k_filter_all_clusters.append(k)

        k_filter = set(k_filter)
        k_new = [k for k in range(lhsnum) if k not in k_filter]
        
        for k in k_new:
            if nonnegs[k]:
                for l in range(w.size):
                    QSP_=QSP(paramarray[k]*(1+(sensitivity_radius*1e-2)*x[l]))
                    dudp=dudp+weights[k]*w[l]*np.dot(QSP_.Sensitivity(),lambda0)

        print(' Writing to file')

        c=csv.writer(open('Data/GlobalSensitivity/'+filename+'sensitivity_steady.csv',"w"))
        c.writerows(dudp)
        del c

    end = time.time()
    print('Run time: ', end - start)
    print('Global sensitivity analysis complete')



Par_list=['\lambda_{T_hM}', '\u03BB_{T_hD}', '\lambda_{T_hH}', '\lambda_{T_hI_{2}}', '\delta_{T_hT_r}','\delta_{T_hIL_{10}}','\delta_{T_h}',\
          '\lambda_{T_cT_h}','\lambda_{T_cI_{2}}', '\lambda_{T_cD}', '\lambda_{T_cI_{\gamma}}','\delta_{T_cIL_{10}}','\delta_{T_CT_r}','\delta_{T_c}',\
          '\lambda_{T_rD}', '\lambda_{T_rI_2}', '\delta_{T_r}',\
          'A_{T_N}','\delta_{T_N}',\
          '\lambda_{MT_h}', '\lambda_{MI_\gamma}','\lambda_{MIL_{10}}','\delta_{M}',\
          'A_{M_N}','\delta_{M_N}',\
          '\lambda_{DH}','\lambda_{DC}','\delta_{DC}','\delta_{D}',\
          'A_{D_N}', '\delta_{D_N}',\
          '\lambda_{C}','\lambda_{CIL_6}','C_0','\delta_{CT_c}','\alpha_{T_c}','\beta_C','\delta_{CI_\gamma}','\delta_{C}',\
          '\u03B1_{NC}','\delta_{N}',\
          '\lambda_{I_\gammaT_c}','\lambda_{I_\gammaT_h}','\lambda_{I_\gammaD}','\delta_{I_\gamma}',\
          '\lambda_{HT_c}', '\lambda_{HT_r}','\lambda_{HT_h}', '\lambda_{HN}','\lambda_{HM}', '\lambda_{HC}','\delta_{H}',\
          '\lambda_{IL_{10}T_h}', '\lambda_{IL_{10}T_c}', '\lambda_{IL_{10}D}', '\lambda_{IL_{10}M}','\delta_{IL_{10}}',\
          '\lambda_{I_2T_c}','\lambda_{I_2T_h}', '\lambda_{I_2D}', '\lambda_{I_2M}','\delta_{I_{2}}',\
          '\lambda_{IL_6C}','\lambda_{IL_6M}','\lambda_{IL_6T_h}','\lambda_{IL_6D}','\delta_{IL_6}']
par_list = ["$"+x+"$" for x in Par_list]

for cluster in range(clusters):
    filename='V67-'+globalgridname+'-'+localgridname+'-cluster-'+str(cluster+1)+'-results-'
    sensitivity_df = pd.read_csv('Data/GlobalSensitivity/'+filename+'sensitivity_steady.csv', header=None)
    sensitivity_df.index = par_list
    sensitive_ids = np.abs(sensitivity_df)[0][abs(sensitivity_df)[0]>0].nlargest(n=20).index
    print('Cluster ', cluster+1)
    print('Sensitivities:\n', sensitivity_df.loc[sensitive_ids])
################################################################################
######################Plotting Sensitivities####################################
usr_inpt2 = input("Do you want to plot the sensitivies?(yes=1, no=0)")

if int(usr_inpt2)==1:
   import matplotlib.pyplot as plt
   import seaborn as sns

   fig, axs = plt.subplots(4, 2, sharey=False, figsize=(11,10))
   fig.subplots_adjust(wspace=0.4, hspace=0.5)
#    sns.set(font_scale=1.1)
   axs[0, 0].set_title('Sensitivity of Cancer', fontsize=14)
   axs[0, 1].set_title('Sensitivity of Total Cells', fontsize=14)

   for cluster in range(clusters):
       filename='V67-'+globalgridname+'-'+localgridname+'-cluster-'+str(cluster+1)+'-results-'
       sensitivity_df = pd.read_csv('Data/GlobalSensitivity/'+filename+'sensitivity_steady.csv', header=None)
       sensitivity_df.index = par_list
       for i in range(2):
           sensitive_ids = np.abs(sensitivity_df)[i][abs(sensitivity_df)[i]>0].nlargest(n=20).index  #filtering singularities and finfing max
           sensitivity_df[i][sensitive_ids[:6]].plot.bar(ax=axs[cluster, i], rot=0, width=0.8)
           axs[cluster, i].axhline()
           axs[cluster, i].set_ylabel('Cluster '+str(cluster+1),fontsize=11)
   plt.savefig('fig/sensitivity1.eps', format='eps',dpi=300)
   plt.rc('xtick',labelsize=14)
   plt.rc('ytick',labelsize=14)

   fig, axs = plt.subplots(4, 2, sharey=False, figsize=(11,10))
   fig.subplots_adjust(wspace=0.4, hspace=0.5)
#    sns.set(font_scale=1.1)
   axs[0, 0].set_title('Sensitivity of Cancer', fontsize=14)
   axs[0, 1].set_title('Sensitivity of Total Cells', fontsize=14)

   for cluster in range(clusters):
       filename='V67-'+globalgridname+'-'+localgridname+'-cluster-'+str(cluster+1)+'-results-'
       sensitivity_df = pd.read_csv('Data/GlobalSensitivity/'+filename+'sensitivity_steady.csv', header=None)
       sensitivity_df.index = par_list
       ids_to_remove = []
       for i in range(2):
           sensitive_ids1 = np.abs(sensitivity_df)[i][abs(sensitivity_df)[i]>0].nlargest(n=20).index  #filtering singularities and finfing max
           sensitive_ids1 = [xx for xx in sensitive_ids1 if xx not in ids_to_remove]
           sensitivity_df[i][sensitive_ids1[6:12]].plot.bar(ax=axs[cluster, i], rot=0, width=0.8)
           axs[cluster, i].axhline()
           axs[cluster, i].set_ylabel('Cluster '+str(cluster+1),fontsize=12)
   plt.savefig('fig/sensitivity2.eps', format='eps',dpi=300)
   plt.rc('xtick', labelsize=14)
   plt.rc('ytick',labelsize=14)
################################################################################
########################Varying parameters######################################
   
def plot_cancer_vary_assumption_perturb_sensitive_params(assumption_idx, assumption_scale, perturb_scale, T):
   import seaborn as sns
   import matplotlib.pyplot as plt
   from matplotlib.lines import Line2D

   restriction_map =  {14:'deltaDC-deltaD', 32:'lambdaIL6Th-lambdaIL6C', 33:'lambdaIL6D-lambdaIL6D'}
   perturb_map = {0: 'no perturbation', -perturb_scale: '-'+str(perturb_scale*10)+'%', perturb_scale: '+'+str(perturb_scale*10)+'%'}
   palette = {0:'#3F9B0B', 1:'#FF796C', 2:'#0343DF',3:'#000000'}
   alphas = [0.15, 0.25, 0.15,0.25]
   restrictions = np.ones(52)

   sns.set(font_scale=1.5)
   sns.set_style("ticks")
   fig, axs = plt.subplots(1, 3, sharey=False, figsize=(17.5,2.5))
   fig.subplots_adjust(wspace=0.4,hspace=0.5)
   axs = axs.flatten()
   t = np.linspace(0, T, 10*T+1)
   custom_lines = [Line2D([0], [0], color='#3F9B0B', lw=3.5),
                   Line2D([0], [0], color='#FF796C', lw=3.5),
                   Line2D([0], [0], color='#0343DF', lw=3.5),
                   Line2D([0], [0], color='#000000', lw=3.5)]
   cancer_vary=pd.DataFrame()
   for i, newscale in enumerate([1, assumption_scale/25, assumption_scale]):
       for cluster in range(4):
           restrictions[assumption_idx] = newscale
           qspcore = Renal_QSP_Functions(SSrestrictions = restrictions)
           new_params = op.fsolve((lambda par,frac: qspcore.SS_system(par,frac)),
                                        np.ones(nparam),args=(clustercells[cluster],))
           
           QSP_ = QSP(new_params)
           u, _ = QSP_.solve_ode(t, IC[cluster], 'given')
           u = clustercells[cluster]*u
           umax = umin = u
           np.savetxt('Cluster'+str(cluster+1)+"Assumption Scale"+str(newscale)+'vary'+restriction_map[assumption_idx]+'.csv', new_params, delimiter=",")
           cancer_vary=pd.concat([cancer_vary,pd.DataFrame(u[:,8],columns=['Cluster '+str(cluster+1)+'Scale '+str(newscale)])], axis=1)
           for param_id in sensitive_param_ids:
               for j in [-perturb_scale, perturb_scale]:
                   perturb_arr = np.zeros(len(new_params))
                   perturb_arr[param_id] = j
                   QSP_ = QSP(new_params*(1+(5e-2)*perturb_arr))
                   u_perturb, _ = QSP_.solve_ode(t, IC[cluster], 'given')
                   u_perturb = clustercells[cluster]*u_perturb
                   umax = np.maximum(u_perturb, umax)
                   umin = np.minimum(u_perturb, umin)
        
           axs[i].margins(x=0)
           axs[i].ticklabel_format(style="sci", axis="y", scilimits=(0, 0))
           axs[i].fill_between(t, umax[:,8], umin[:,8], facecolor=palette[cluster], alpha=alphas[cluster])
           axs[i].plot(t, u[:,8], color=palette[cluster])
       axs[i].set_xlabel('time (days)',fontsize=14)
       axs[i].set_ylabel('Cancer cells',fontsize=14)
       axs[i].set_title('Scale='+str(newscale),fontsize=14)
   if assumption_idx==32:
       axs[2].legend(custom_lines,['Cluster 1', 'Cluster 2', 'Cluster 3', 'Cluster 4'], bbox_to_anchor=(1.1, 0.5),loc='center left')
       plt.savefig('fig/varying_Scale'+str(cluster+1)+restriction_map[assumption_idx]+'.eps', format='eps',dpi=300)
   plt.rc('xtick',labelsize=14)
   plt.rc('ytick',labelsize=14)
   plt.show()

usr_inpt4 = input("Do you want to plot the varying dynamics?(yes=1, no=0)")
restriction_map = {14:'deltaDC-deltaD', 32:'lambdaIL6Th-lambdaIL6C', 33:'lambdaIL6D-lambdaIL6D'}
keys= [14,32,33]
sensitive_param_ids = [37,34,38,22,24,19,20,21,15,14,65,62,16]
if int(usr_inpt4)==1:
   for i in range(len(keys)):
       print('Varying', restriction_map[keys[i]], 'assumption + perturb sensitive params by 5%')
       plot_cancer_vary_assumption_perturb_sensitive_params(assumption_idx=keys[i], assumption_scale=5, perturb_scale=1, T=5000)


#######################Plot varying dynamics by scaling Alpha_Tc####################      
def plot_cancer_vary_PD_PDL(assumption_idx, assumption_scale, perturb_scale, T):
   import seaborn as sns
   import matplotlib.pyplot as plt
   from matplotlib.lines import Line2D

   restriction_map = {16:'deltaCTc-deltaCIgamma', 15: 'deltaC-deltaCTc', 34:'deltaMN-deltaM',11:'lambdaMTh-lambdaMIgamma', 12:'lambdaMIgamma-lambdaMIL10',\
                        10:'lambdaTrD-lambdaTrI2',  33:'lambdaIL6D-lambdaIL6D',41:'deltaTr'}
   perturb_map = {0: 'no perturbation', -perturb_scale: '-'+str(perturb_scale*10)+'%', perturb_scale: '+'+str(perturb_scale*10)+'%'}
   palette = {0:'#3F9B0B', 1:'#FF796C', 2:'#0343DF',3:'#000000'}
   alphas = [0.15, 0.25, 0.15,0.25]
   restrictions = np.ones(52)

   sns.set(font_scale=1.5)
   sns.set_style("ticks")
   fig, axs = plt.subplots(1, 3, sharey=False, figsize=(17.5,2.5))
   fig.subplots_adjust(wspace=0.4,hspace=0.5)
   axs = axs.flatten()
   t = np.linspace(0, T, 10*T+1)
   custom_lines = [Line2D([0], [0], color='#3F9B0B', lw=3.5),
                   Line2D([0], [0], color='#FF796C', lw=3.5),
                   Line2D([0], [0], color='#0343DF', lw=3.5),
                   Line2D([0], [0], color='#000000', lw=3.5)]
   cancer_vary=pd.DataFrame()
   for i, newscale in enumerate([1, assumption_scale/25, assumption_scale]):
       for cluster in range(4):
           restrictions[assumption_idx] = newscale
           qspcore = Renal_QSP_Functions(SSrestrictions = restrictions)
           new_params = op.fsolve((lambda par,frac: qspcore.SS_system(par,frac)),
                                        np.ones(nparam),args=(clustercells[cluster],))
           restrictions[37] = newscale
           qspcore = Renal_QSP_Functions(SSrestrictions = restrictions)
           params = op.fsolve((lambda par,frac: qspcore.SS_system(par,frac)), np.ones(nparam),args=(clustercells[cluster],))
           QSP_ = QSP(new_params)
           u, _ = QSP_.solve_ode(t, IC[cluster], 'given')
           u = clustercells[cluster]*u
           umax = umin = u
           #np.savetxt('Cluster'+str(cluster+1)+restriction_map[assumption_idx]+"Assumption Scale"+str(newscale)+'.csv', new_params, delimiter=",")
           cancer_vary=pd.concat([cancer_vary,pd.DataFrame(u[:,8],columns=['Cluster '+str(cluster+1)+'Scale '+str(newscale)])], axis=1)
           for param_id in sensitive_param_ids:
               for j in [-perturb_scale, perturb_scale]:
                   perturb_arr = np.zeros(len(new_params))
                   perturb_arr[param_id] = j
                   QSP_ = QSP(new_params*(1+(5e-1)*perturb_arr))
                   u_perturb, _ = QSP_.solve_ode(t, IC[cluster], 'given')
                   u_perturb = clustercells[cluster]*u_perturb
                   umax = np.maximum(u_perturb, umax)
                   umin = np.minimum(u_perturb, umin)
        
           axs[i].margins(x=0)
           axs[i].ticklabel_format(style="sci", axis="y", scilimits=(0, 0))
           axs[i].fill_between(t, umax[:,8], umin[:,8], facecolor=palette[cluster], alpha=alphas[cluster])
           axs[i].plot(t, u[:,8], color=palette[cluster])
       axs[i].set_xlabel('time (days)',fontsize=14)
       axs[i].set_ylabel('Cancer cells',fontsize=14)
       axs[i].set_title('Scale='+str(newscale),fontsize=14)
   axs[2].legend(custom_lines,['Cluster 1', 'Cluster 2', 'Cluster 3', 'Cluster 4'], bbox_to_anchor=(1.1, 0.5),loc='center left')
   plt.rc('xtick',labelsize=14)
   plt.rc('ytick',labelsize=14)
   plt.savefig('fig/VaryDyn_VaryAlphaTc.eps',format='eps', bbox_inches='tight' )
   plt.show()

usr_inpt5 = input("Do you want to plot the varying dynamics with Alpha_Tc scaling?(yes=1, no=0)")
restriction_map = {37:'alphaTc'}
keys= [37]
sensitive_param_ids = [35]
if int(usr_inpt5)==1:
   for i in range(len(keys)):
       print('Varying', restriction_map[keys[i]], 'assumption + perturb sensitive params by 5%')
       plot_cancer_vary_PD_PDL(assumption_idx=keys[i], assumption_scale=5, perturb_scale=1, T=5000)


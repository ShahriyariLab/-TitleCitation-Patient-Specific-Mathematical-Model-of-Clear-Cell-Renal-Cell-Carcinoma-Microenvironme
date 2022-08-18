#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 26 09:54:59 2022

@author: dilruba
"""


## from odesolver import *
import numpy as np
import scipy.optimize as op
from itertools import permutations
from scipy.integrate import odeint
import pandas as pd

def Sensitivity_function(x,t,QSP):
    nparam=QSP.qspcore.nparam
    nvar=QSP.qspcore.nvar
    x=x.reshape(nparam+1,nvar)
    dxdt=np.empty(x.shape)
    dxdt[0]=QSP(x[0],t)
    dxdt[1:]=np.dot(QSP.Ju(x[0],t),x[1:].T).T+ QSP.Jp(x[0],t)
    return dxdt.flatten()
# Object class that defines the functions for the appropriate QSP Model
class Renal_QSP_Functions(object):
    def __init__(self,SSrestrictions=np.ones(52)):
        self.nparam=67
        self.nvar=15
        self.SSscale = SSrestrictions
        self.variable_names=[ '$T_h$', '$T_C$', '$T_r$', 'Naive $T$','$M\phi$', 'Naive $M\phi$', '$DC$', 'Naive $DC$', 'Cancer', 'Necrotic','$IFN_\gamma$','$HMGB1$','$IL-10$', '$I-2$', '$IL-6$']
        self.parameter_names=['\lambda_{T_hM}', '\lamda_{T_hD}', '\lambda_{T_hH}', '\lambda_{T_hI_{2}}', '\delta_{T_hT_r}','\delta_{T_hIL_{10}}','\delta_{T_h}',\
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
    def __call__(self,x,t,p):
        # ODE right-hand side
        dx0=(p[0]*x[4]+p[1]*x[6]+p[2]*x[11]+p[3]*x[13])*x[3]-(p[4]*x[2]+p[5]*x[12]+p[6])*x[0]
        dx1=(p[7]*x[0]+p[8]*x[13]+p[9]*x[6]+p[10]*x[10])*x[3]-(p[11]*x[12]+p[12]*x[2]+p[13])*x[1]
        dx2=(p[14]*x[6]+p[15]*x[13])*x[3]-p[16]*x[2]
        dx3=p[17]-(p[0]*x[4]+p[1]*x[6]+p[2]*x[11]+p[3]*x[13])*x[3]-(p[7]*x[0]+p[8]*x[13]+p[9]*x[6]+p[10]*x[10])*x[3]-(p[14]*x[6]+p[15]*x[13])*x[3]-p[18]*x[3]
        dx4=(p[19]*x[0]+p[20]*x[10]+p[21]*x[12])*x[5]-p[22]*x[4]
        dx5=p[23]-(p[19]*x[0]+p[20]*x[10]+p[21]*x[12])*x[5]-p[24]*x[5]
        dx6=(p[25]*x[11]+p[26]*x[8])*x[7]-(p[27]*x[8]+p[28])*x[6]
        dx7= p[29]-(p[25]*x[11]+p[26]*x[8])*x[7]-p[30]*x[7]
        dx8=(p[31]+p[32]*x[14])*(1-x[8]/p[33])*x[8]-(p[34]*x[1]*(1/(1+p[35]*x[1]*p[36]*x[8]))+p[37]*x[10]+p[38])*x[8]
        dx9=p[39]*(p[34]*x[1]*(1/(1+p[35]*x[1]*p[36]*x[8]))+p[37]*x[10]+p[38])*x[8]-p[40]*x[9]
        dx10=(p[41]*x[1]+p[42]*x[0]+p[43]*x[6])-p[44]*x[10]
        dx11=p[45]*x[1]+p[46]*x[2]+p[47]*x[0]+p[48]*x[9]+p[49]*x[4]+p[50]*x[8]-p[51]*x[11]
        dx12=p[52]*x[0]+p[53]*x[1]+p[54]*x[6]+p[55]*x[4]-p[56]*x[12]
        dx13=p[57]*x[1]+p[58]*x[0]+p[59]*x[6]+p[60]*x[4]-p[61]*x[13]
        dx14=p[62]*x[8]+p[63]*x[4]+p[64]*x[0]+p[65]*x[6]-p[66]*x[14]
        dx=np.array([dx0,dx1,dx2,dx3,dx4,dx5,dx6,dx7,dx8,dx9,dx10,dx11,dx12,dx13,dx14])
        return dx
    def Ju(self,x,t,p):
        # Jacobian with respect to variables
        return np.array([[-p[6] - p[5]*x[12] - p[4]*x[2], 0, -p[4]*x[0], p[2]*x[11] + p[3]*x[13] + p[0]*x[4] + p[1]*x[6], p[0]*x[3], 0, p[1]*x[3], 0, 0, 0, 0, p[2]*x[3], -p[5]*x[0], p[3]*x[3], 0],\
                         [p[7]*x[3], -p[13] - p[11]*x[12] - p[12]*x[2], -p[12]*x[1], p[7]*x[0] + p[10]*x[10] + p[8]*x[13] + p[9]*x[6], 0, 0, p[9]*x[3], 0, 0, 0, p[10]*x[3], 0, -p[11]*x[1], p[8]*x[3], 0],\
                         [0, 0, -p[16], p[15]*x[13] + p[14]*x[6], 0, 0, p[14]*x[3], 0, 0, 0, 0, 0, 0, p[15]*x[3], 0],\
                         [-p[7]*x[3], 0, 0, -p[18] - p[7]*x[0] - p[10]*x[10] - p[2]*x[11] - p[15]*x[13] - p[3]*x[13] - p[8]*x[13] - p[0]*x[4] - p[1]*x[6] - p[14]*x[6] - p[9]*x[6], -p[0]*x[3], 0, -p[1]*x[3] - p[14]*x[3] - p[9]*x[3], 0, 0, 0, -p[10]*x[3], -p[2]*x[3], 0, -p[15]*x[3] - p[3]*x[3] - p[8]*x[3], 0],\
                         [p[19]*x[5], 0, 0, 0, -p[22], p[19]*x[0] + p[20]*x[10] + p[21]*x[12], 0, 0, 0, 0, p[20]*x[5], 0, p[21]*x[5], 0, 0],\
                         [-p[19]*x[5], 0, 0, 0, 0, -p[24] - p[19]*x[0] - p[20]*x[10] - p[21]*x[12], 0, 0, 0, 0, -p[20]*x[5], 0, -p[21]*x[5], 0, 0],\
                         [0, 0, 0, 0, 0, 0, -p[28] - p[27]*x[8], p[25]*x[11] + p[26]*x[8], -p[27]*x[6] + p[26]*x[7], 0, 0, p[25]*x[7], 0, 0, 0],\
                         [0, 0, 0, 0, 0, 0, 0, -p[30] - p[25]*x[11] - p[26]*x[8], -p[26]*x[7], 0, 0, -p[25]*x[7], 0, 0, 0],\
                         [0, x[8]*((p[34]*p[35]*p[36]*x[1]*x[8])/((1 + p[35]*p[36]*x[1]*x[8])**2) - p[34]/(1 + p[35]*p[36]*x[1]*x[8])), 0, 0, 0, 0, 0, 0, -p[38] - p[37]*x[10] - (p[31] + p[32]*x[14])*x[8]/p[33] + (p[31] + p[32]*x[14])*(1 - x[8]/p[33]) + (p[34]*p[35]*p[36]*x[1]**2*x[8])/((1 + p[35]*p[36]*x[1]*x[8])**2) - p[34]*x[1]/(1 + p[35]*p[36]*x[1]*x[8]), 0, -p[37]*x[8], 0, 0, 0, p[32]*x[8]*(1 - x[8]/p[33])],\
                         [0, p[39]*x[8]*(-(p[34]*p[35]*p[36]*x[1]*x[8])/((1 + p[35]*p[36]*x[1]*x[8])**2) + p[34]/(1 + p[35]*p[36]*x[1]*x[8])), 0, 0, 0, 0, 0, 0, (-p[34]*p[35]*p[36]*p[39]*x[1]**2*x[8])/((1 + p[35]*p[36]*x[1]*x[8])**2) + p[39]*(p[38] + p[37]*x[10] + p[34]*x[1]/(1 + p[35]*p[36]*x[1]*x[8])), -p[40], p[37]*p[39]*x[8], 0, 0, 0, 0],\
                         [p[42], p[41], 0, 0, 0, 0, p[43], 0, 0, 0, -p[44], 0, 0, 0, 0],\
                         [p[47], p[45], p[46], 0, p[49], 0, 0, 0, p[50], p[48], 0, -p[51], 0, 0, 0],\
                         [p[52], p[53], 0, 0, p[55], 0, p[54], 0, 0, 0, 0, 0, -p[56], 0, 0],\
                         [p[58], p[57], 0, 0, p[60], 0, p[59], 0, 0, 0, 0, 0, 0, -p[61], 0],\
                         [p[64], 0, 0, 0, p[63], 0, p[65], 0, p[62], 0, 0, 0, 0, 0, -p[66]]])
    def Jp(self,x,t,p):
        #jacobian with respect to each of the model
        return np.array([[x[3]*x[4], 0, 0, -x[3]*x[4], 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                         [x[3]*x[6], 0, 0, -x[3]*x[6], 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                         [x[3]*x[11], 0, 0, -x[3]*x[11], 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                         [x[3]*x[13], 0, 0, -x[3]*x[13], 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                         [-x[0]*x[2], 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                         [-x[0]*x[12], 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                         [-x[0], 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                         [0, x[0]*x[3], 0, -x[0]*x[3], 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                         [0, x[3]*x[13], 0, -x[3]*x[13], 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                         [0, x[3]*x[6], 0, -x[3]*x[6], 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                         [0, x[3]*x[10], 0, -x[3]*x[10], 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                         [0, -x[1]*x[12], 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                         [0, -x[1]*x[2], 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                         [0, -x[1], 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                         [0, 0, x[3]*x[6], -x[3]*x[6], 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                         [0, 0, x[3]*x[13], -x[3]*x[13], 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                         [0, 0, -x[2], 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                         [0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                         [0, 0, 0, -x[3], 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                         [0, 0, 0, 0, x[0]*x[5], -x[0]*x[5], 0, 0, 0, 0, 0, 0, 0, 0, 0],
                         [0, 0, 0, 0, x[5]*x[10], -x[5]*x[10], 0, 0, 0, 0, 0, 0, 0, 0, 0],
                         [0, 0, 0, 0, x[5]*x[12], -x[5]*x[12], 0, 0, 0, 0, 0, 0, 0, 0, 0],
                         [0, 0, 0, 0, -x[4], 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                         [0, 0, 0, 0, 0, 1,0, 0, 0, 0, 0, 0, 0, 0, 0],
                         [0, 0, 0, 0, 0, -x[5], 0, 0, 0, 0, 0, 0, 0, 0, 0],
                         [0, 0, 0, 0, 0, 0, x[7]*x[11], -x[7]*x[11], 0, 0, 0, 0, 0, 0, 0],
                         [0, 0, 0, 0, 0, 0, x[7]*x[8], -x[7]*x[8], 0, 0, 0, 0, 0, 0, 0],
                         [0, 0, 0, 0, 0, 0, -x[6]*x[8], 0, 0, 0, 0, 0, 0, 0, 0],
                         [0, 0, 0, 0, 0, 0, -x[6], 0, 0, 0, 0, 0, 0, 0, 0],
                         [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
                         [0, 0, 0, 0, 0, 0, 0, -x[7], 0, 0, 0, 0, 0, 0, 0],
                         [0, 0, 0, 0, 0, 0, 0, 0, x[8]*(1 - x[8]/p[33]), 0, 0, 0, 0, 0, 0],
                         [0, 0, 0, 0, 0, 0, 0, 0, x[8]*x[14]*(1 - x[8]/p[33]), 0, 0, 0, 0, 0, 0],
                         [0, 0, 0, 0, 0, 0, 0, 0, x[8]**2*(p[31] + p[32]*x[14])/p[33]**2, 0, 0, 0, 0, 0, 0],
                         [0, 0, 0, 0, 0, 0, 0, 0, -x[1]*x[8]/(1 + p[35]*p[36]*x[1]*x[8]), p[39]*x[1]*x[8]/(1 + p[35]*p[36]*x[1]*x[8]), 0, 0, 0, 0, 0],
                         [0, 0, 0, 0, 0, 0, 0, 0, p[34]*p[36]*x[1]**2*x[8]**2/((1 + p[35]*p[36]*x[1]*x[8])**2), -p[34]*p[36]*p[39]*x[1]**2*x[8]**2/((1 + p[35]*p[36]*x[1]*x[8])**2), 0, 0, 0, 0, 0],\
                         [0, 0, 0, 0, 0, 0, 0, 0, p[34]*p[35]*x[1]**2*x[8]**2/((1 + p[35]*p[36]*x[1]*x[8])**2), -p[34]*p[35]*p[39]*x[1]**2*x[8]**2/((1 + p[35]*p[36]*x[1]*x[8])**2), 0, 0, 0, 0, 0],\
                         [0, 0, 0, 0, 0, 0, 0, 0, -x[8]*x[10], p[39]*x[8]*x[10], 0, 0, 0, 0, 0],
                         [0, 0, 0, 0, 0, 0, 0, 0, -x[8], p[39]*x[8], 0, 0, 0, 0, 0],
                         [0, 0, 0, 0, 0, 0, 0, 0, 0, x[8]*(p[38] + p[34]*x[1]/(1 + p[35]*p[36]*x[1]*x[8]) + p[37]*x[10]), 0, 0, 0, 0, 0],\
                         [0, 0, 0, 0, 0, 0, 0, 0, 0, -x[9],0, 0, 0, 0, 0],\
                         [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, x[1], 0, 0, 0, 0],
                         [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, x[0], 0, 0, 0, 0],
                         [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, x[6], 0, 0, 0, 0],
                         [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -x[10], 0, 0, 0, 0],
                         [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, x[1], 0, 0, 0],
                         [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, x[2], 0, 0, 0],
                         [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, x[0], 0, 0, 0],
                         [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, x[9], 0, 0, 0],
                         [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, x[4], 0, 0, 0],
                         [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, x[8], 0, 0, 0],
                         [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -x[11], 0, 0, 0],
                         [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,0, x[0], 0, 0],
                         [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, x[1], 0, 0],
                         [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, x[6], 0, 0],
                         [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, x[4], 0, 0],
                         [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -x[12], 0, 0],
                         [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, x[1], 0],
                         [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, x[0], 0],
                         [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, x[6], 0],
                         [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, x[4], 0],
                         [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -x[13], 0],
                         [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, x[8]],
                         [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, x[4]],
                         [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, x[0]],
                         [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, x[6]],
                         [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -x[14]]])
    def SS_system(self,p,frac):
         # compute the system and restrictions with non-dimensional steady states at 1
         # pre-defined rates, extreme and global mean values are hardcoded here
         # cell fractions are given as [Tn Th Tc Tr Dn D M M0 C N]
         #in case of non-dimensional use 1's
         x=np.ones(self.nvar);
         #in case of dimensional use:
         #x = frac
         # rates acquired from bio research =p[6]=p[13],
         # [delta_{t_h},delta_{T_c},delta_{T_r},delta_{T_N},delta_{M},delta_D,delta_{I_gamma},delta_H,delta_{IL_{10}},delta_{I_2},delta_{IL_6}]
         # =[p[6],p[13],p[16],p[18],p[22],p[28],p[44],p[51],p[56],p[61],p[66]]
         globvals=np.array([0.231,0.406,0.231,0.000949,0.0198,0.277,33.3,18,4.62,2.238,1.07])
         # maximal values of each variable across all patients [ \"Th\", \"Tc\", \"Tr\",\"Tn\",\"M\", \"Mn\",  \"D\", \"Dn\",  \"C\", \"N\", \"Ig\", \"H\",\"IL10\", \"I2\",  \"IL6\"]
         extremevals=pd.read_csv('input/extvals_dim_6_3.csv').to_numpy()[0]
         # average values of each variable across all patients [Tc mu1 Ig Gb]
         meanvals=pd.read_csv('input/meanvals_dim_6_3.csv').to_numpy()[0]
        
         #System 
         #right hand side of the equation with added parameter scales that can changed for sensitivity analysis
         sx0=(p[0]*x[4]+p[1]*x[6]+p[2]*x[11]+p[3]*x[13])*x[3]-(p[4]*x[2]+p[5]*x[12]+p[6])*x[0]
         sx1=(p[7]*x[0]+p[8]*x[13]+p[9]*x[6]+p[10]*x[10])*x[3]-(p[11]*x[12]+p[12]*x[2]+p[13])*x[1]
         sx2=(p[14]*x[6]+p[15]*x[13])*x[3]-p[16]*x[2]
         sx3=p[17]-(p[0]*x[4]+p[1]*x[6]+p[2]*x[11]+p[3]*x[13])*x[3]-(p[7]*x[0]+p[8]*x[13]+p[9]*x[6]+p[10]*x[10])*x[3]-(p[14]*x[6]+p[15]*x[13])*x[3]-p[18]*x[3]
         sx4=(p[19]*x[0]+p[20]*x[10]+p[21]*x[12])*x[5]-p[22]*x[4]
         sx5=p[23]-(p[19]*x[0]+p[20]*x[10]+p[21]*x[12])*x[5]-p[24]*x[5]
         sx6=(p[25]*x[11]+p[26]*x[8])*x[7]-(p[27]*x[8]+p[28])*x[6]
         sx7= p[29]-(p[25]*x[11]+p[26]*x[8])*x[7]-p[30]*x[7]
         sx8=(p[31]+p[32]*x[14])*(1-x[8]/p[33])*x[8]-(p[34]*x[1]*(1/(1+p[35]*x[1]*p[36]*x[8]))+p[37]*x[10]+p[38])*x[8]
         sx9=p[39]*(p[34]*x[1]*(1/(1+p[35]*x[1]*p[36]*x[8]))+p[37]*x[10]+p[38])*x[8]-p[40]*x[9]
         sx10=(p[41]*x[1]+p[42]*x[0]+p[43]*x[6])-p[44]*x[10]
         sx11=p[45]*x[1]+p[46]*x[2]+p[47]*x[0]+p[48]*x[9]+p[49]*x[4]+p[50]*x[8]-p[51]*x[11]
         sx12=p[52]*x[0]+p[53]*x[1]+p[54]*x[6]+p[55]*x[4]-p[56]*x[12]
         sx13=p[57]*x[1]+p[58]*x[0]+p[59]*x[6]+p[60]*x[4]-p[61]*x[13]
         sx14=p[62]*x[8]+p[63]*x[4]+p[64]*x[0]+p[65]*x[6]-p[66]*x[14]
         #assumptions in relations to known parameter values
         a0=self.SSscale[0]*p[0] - p[1]*(extremevals[6]/frac[6])/(extremevals[4]/frac[4])
         a1=self.SSscale[1]*p[2] -p[1]*(extremevals[6]/frac[6])/(200*(extremevals[11]/frac[11]))
         a2=self.SSscale[2]*p[3] - p[1]*(extremevals[6]/frac[6])/(200*(extremevals[13]/frac[13]))
         a3=self.SSscale[3]*p[4] - 20*p[6]/(extremevals[2]/frac[2])
         a4=self.SSscale[4]*p[5] - 20*p[6]/(extremevals[12]/frac[12]) #error catched in mathematica code
         a5=self.SSscale[5]*p[8] - p[7]*(extremevals[0]/frac[0])/(extremevals[13]/frac[13])
         a6=self.SSscale[6]*p[9] - p[7]*(extremevals[0]/frac[0])/(extremevals[6]/frac[6])
         a7=self.SSscale[7]*p[10] - p[7]*(extremevals[0]/frac[0])/(100*extremevals[10]/frac[10])
         a8=self.SSscale[8]*p[11] - 20*p[13]/(extremevals[12]/frac[12])
         a9=self.SSscale[9]*p[12] - 20*p[13]/(extremevals[2]/frac[2])
         a10=self.SSscale[10]*p[14] - 100*p[15]*(extremevals[13]/frac[13])/(extremevals[6]/frac[6])
         a11=self.SSscale[11]*p[19] - p[20]*(extremevals[10]/frac[10])/(extremevals[0]/frac[0])
         a12=self.SSscale[12]*p[21] -p[20]*(extremevals[10]/frac[10])/(extremevals[12]/frac[12])
         a14=self.SSscale[13]*p[25] - p[26]*(extremevals[8]/frac[8])/(extremevals[11]/frac[11])
         a15=self.SSscale[14]*p[27] - 50*p[28]/(extremevals[8]/frac[8])
         a17=self.SSscale[15]*p[38] - p[34]*meanvals[1]
         a19=self.SSscale[16]*p[34] - p[37]*(meanvals[10]/frac[10])/(meanvals[1]/frac[1])
         a22=self.SSscale[17]*p[39] - 0.5*(frac[8]/frac[9])
         a23=self.SSscale[18]*p[41] - 5*p[42]*(extremevals[0]/frac[0])/(extremevals[1]/frac[1])
         a24=self.SSscale[19]*p[43] - 5*p[42]*(extremevals[0]/frac[0])/(extremevals[6]/frac[6])
         a25=self.SSscale[20]*p[45] - p[50]*(extremevals[8]/frac[8])/(10*extremevals[1]/frac[1])
         a26=self.SSscale[21]*p[46] - p[50]*(extremevals[8]/frac[8])/(10*extremevals[2]/frac[2])
         a27=self.SSscale[22]*p[47] - p[50]*(extremevals[8]/frac[8])/(10*extremevals[0]/frac[0])
         a28=self.SSscale[23]*p[48] - p[50]*(extremevals[8]/frac[8])/(extremevals[9]/frac[9])
         a29=self.SSscale[24]*p[49] - p[50]*(extremevals[8]/frac[8])/(10*extremevals[4]/frac[4])
         a30=self.SSscale[25]*p[53] - p[52]*(extremevals[0]/frac[0])/(extremevals[1]/frac[1])
         a31=self.SSscale[26]*p[54] - p[52]*(extremevals[0]/frac[0])/(extremevals[6]/frac[6])
         a32=self.SSscale[27]*p[55] - p[52]*(extremevals[0]/frac[0])/(extremevals[4]/frac[4])
         a33=self.SSscale[28]*p[57] - p[58]*(extremevals[0]/frac[0])/(2*extremevals[1]/frac[1])
         a34=self.SSscale[29]*p[59] - p[58]*(extremevals[0]/frac[0])/(2*extremevals[6]/frac[6])
         a35=self.SSscale[30]*p[60] - p[58]*(extremevals[0]/frac[0])/(2*extremevals[4]/frac[4])
         a36=self.SSscale[31]*p[63] - p[62]*(extremevals[8]/frac[8])/(1.5*extremevals[4]/frac[4])
         a37=self.SSscale[32]*p[64] - p[62]*(extremevals[8]/frac[8])/(1.5*extremevals[0]/frac[0])
         a38=self.SSscale[33]*p[65] - p[62]*(extremevals[8]/frac[8])/(1.5*extremevals[6]/frac[6])
         a13=self.SSscale[34]*p[24] - p[22]
         a16=self.SSscale[35]*p[30] - p[28]
         a18=self.SSscale[36]*p[33] - 2.5*extremevals[8]/frac[8] #kept it same as mathematica
         a20=self.SSscale[37]*p[35] - (8.9*10**(-5))*extremevals[1]/frac[1] 
         a21=self.SSscale[38]*p[36] - (3.3*10**(-5))*extremevals[8]/frac[8]
         #known parameters
         a39=self.SSscale[39]*p[6]-globvals[0]
         a40=self.SSscale[40]*p[13]-globvals[1]
         a41=self.SSscale[41]*p[16]-globvals[2]
         a42=self.SSscale[42]*p[18]-globvals[3]
         a43=self.SSscale[43]*p[22]-globvals[4]
         a45=self.SSscale[44]*p[28]-globvals[5]
         a46=self.SSscale[45]*p[44]-globvals[6]
         a47=self.SSscale[46]*p[51]-globvals[7]
         a48=self.SSscale[47]*p[56]-globvals[8]
         a49=self.SSscale[48]*p[61]-globvals[9]
         a50=self.SSscale[49]*p[66]-globvals[10]
         #Cancer cell rate relations
         a51=(p[31] + p[32]*meanvals[14]/frac[14])-p[38]-self.SSscale[50]*np.log(2)/278 #different from mathematica
         a52=p[31]-self.SSscale[51]*np.log(2)/642-(p[34]*meanvals[1]/frac[1]*(1/(1 + p[35]*(meanvals[1]/frac[1])*p[36]*(meanvals[8]/frac[8]))) + p[37]*meanvals[10]/frac[10] + p[38]) #different from mathematica
         #the whole system as an array
         system=np.array([sx0,sx1,sx2,sx3,sx4,sx5,sx6,sx7,sx8,sx9,sx10,sx11,sx12,sx13,sx14,a0,a1,a2,a3,a4,a5,a6,a7,a8,a9,a10,a11,a12,a13,a14,a15,a16,a17,a18,a19,a20,a21,a22,a23,a24,a25,a26,a27,a28,a29,a30,a31,a32,a33,a34,a35,a36,a37,a38,a39,a40,a41,a42,a43,a45,a46,a47,a48,a49,a50,a51,a52])
         return system

class QSP:
    def __init__(self,parameters,qspcore=Renal_QSP_Functions()):
        self.qspcore=qspcore
        self.p=parameters;
    def set_parameters(self,parameters):
        self.p=parameters;
    def steady_state(self):
        # compute steady state with current parameters
        IC=np.ones(self.qspcore.nvar);
        return op.fsolve((lambda x: self.qspcore(x,0,self.p)),IC,fprime=(lambda x: self.qspcore.Ju(x,0,self.p)),xtol=1e-7,maxfev=1000)  #This might need to change
    def Sensitivity(self,method='steady',t=None,IC=None,params=None,variables=None):
        # Sensitivity matrix
        # method: (default) 'steady' - steady state sensitivity
                # 'time' - time-integrated sensitivity
                        # requires time array t and initial conditions IC
                # 'split' - steady state sensitivity with respect to chosen parameters
                        # requires initiate_parameter_split to have been run
                        # takes optional argument 'variables' for sensitivity of specific variables.
        if method=='time':
            if IC is None:
                raise Exception('Error: Need initial conditions for time integration. Set IC=')
                return None
            if t is None:
                raise Exception('Error: Need time values for time integration. Set t=')
                return None

            nparam=self.qspcore.nparam
            nvar=self.qspcore.nvar
            initial=np.zeros((nparam+1,nvar)); #changed +1 from the original one to the total number of parameters; kept it same
            return np.mean(odeint(Sensitivity_function, initial.flatten(), t, args=(self, )) ,axis=0).reshape(nparam+1,nvar)[1:]
        elif method=='split':
            if not hasattr(self,'variable_par'):
                raise Exception('error: parameter splitting is not set. use "initiate_parameter_split" method')
                return None
            if params is None:
                raise Exception('error: Need parameter values for split sensitivity. Set params=')
                return None
            elif len(params)!=sum(self.variable_par):  #what is variable_par?
                raise Exception('error: wrong number of parameters given')
                return None

            if IC is None:
                IC=np.ones(self.qspcore.nvar);
            par=np.copy(self.p)
            par[self.variable_par]=np.copy(params)

            u=op.fsolve((lambda x: self.qspcore(x,0,par)),IC,fprime=(lambda x: self.qspcore.Ju(x,0,par)),xtol=1e-7,maxfev=1000)
            if variables is None:
                return -np.dot(self.qspcore.Jp(u,0,self.p),np.linalg.inv(self.qspcore.Ju(u,0,self.p).T))[self.variable_par]
            else:
                return -np.dot(self.qspcore.Jp(u,0,self.p),np.linalg.inv(self.qspcore.Ju(u,0,self.p).T))[self.variable_par,variables]
        else:
            u=self.steady_state()
            return -np.dot(self.qspcore.Jp(u,0,self.p),np.linalg.inv(self.qspcore.Ju(u,0,self.p).T))
    def __call__(self,x,t):
        return self.qspcore(x,t,self.p)
    def Ju(self,x,t):
        return self.qspcore.Ju(x,t,self.p)
    def Jp(self,x,t):
        return self.qspcore.Jp(x,t,self.p)
    def variable_names(self):return self.qspcore.variable_names
    def parameter_names(self):return self.qspcore.parameter_names
    def solve_ode(self, t, IC, method='default'):
        # Solve ode system with either default 1e4 time steps or given time discretization
        # t - time: for 'default' needs start and end time
        #           for 'given' needs full array of time discretization points
        # IC - initial conditions
        # method: 'default' - given interval divided by 10000 time steps
        #         'given' - given time discretization
        if method=='given':
            return odeint((lambda x,t: self.qspcore(x,t,self.p)), IC, t,
                            Dfun=(lambda x,t: self.qspcore.Ju(x,t,self.p))), t
        else:
            return odeint((lambda x,t: self.qspcore(x,t,self.p)), IC, np.linspace(min(t), max(t), 5001),
                            Dfun=(lambda x,t: self.qspcore.Ju(x,t,self.p))), np.linspace(min(t), max(t), 5001)

    def initiate_parameter_split(self,variable_par):
        # splits the parameters into fixed and variable for further fittin
        # variable_par - boolean array same size as parameter array indicating which parameters are variable
        if (variable_par.dtype!='bool') or (len(variable_par)!=self.qspcore.nparam):
            raise Exception('error: wrong parameter indicator')
            return None
        self.variable_par=np.copy(variable_par)

    def solve_ode_split(self, t, IC, params):
        # Solve ode system with adjusted variable parameters
        #   using either default 1e4 time steps or given time discretization
        # t - time: needs full array of time discretization points
        # IC - initial conditions
        # params - parameters to update for this solution
        if not hasattr(self,'variable_par'):
            raise Exception('error: parameter splitting is not set. use "initiate_parameter_split" method')
            return None
        if len(params)!=sum(self.variable_par):
            raise Exception('error: wrong number of parameters given')
            return None
        par=np.copy(self.p)
        par[self.variable_par]=np.copy(params)
        return odeint((lambda x,t: self.qspcore(x,t,par)), IC, t,
                            Dfun=(lambda x,t: self.qspcore.Ju(x,t,par)))

#    def Bifurcation(self,IC):
#        return op.fsolve((lambda x: self.qspcore(x,0,self.p)),IC,fprime=(lambda x: self.qspcore.Ju(x,0,self.p)),xtol=1e-7,maxfev=10000)

    @classmethod
    def from_cell_data(class_object, fracs, qspcore=Renal_QSP_Functions()):
        params=op.fsolve((lambda p,fracs: qspcore.SS_system(p,fracs)),np.ones(qspcore.nparam),
                         args=(fracs,))
        return class_object(params)
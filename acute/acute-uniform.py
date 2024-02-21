from brian2 import *
import numpy as np
import matplotlib.pyplot as plt
import random
from scipy.stats import pearsonr
from scipy.optimize import curve_fit
from sklearn.decomposition import PCA
import pandas as pd
import re
from brian2 import Experiment
plt.switch_backend('agg')

regx=re.compile(r'\d+.\d+')


def get_population(name,N,tau_rp,g_m,C_m,g_AMPA_ext,g_GABA,g_AMPA_rec,g_NMDA,I_background,lamda_NA,tau_GABA):
    """Get population of neurons

       name -- name of population
       tau_rp -- refractory period
       g_m -- diandao of neurons
       C_m -- capacitance of neurons
       """
    V_thr = -50. * mV
    V_reset = -55. * mV
    V_L = -70. * mV
    V_I = -70. * mV
    
    Mg2 = 1.
    c1=120 * pA
    c2=136.24 *pA
    c3=7.0
    c4=0 *pA
    c5=9.64 *pA
    c6=20 *pA
    c7=1*pA
    mu_e=1.5
    mu_i=2.24
    neuron_spacing = 50*umetre
    width = N/4.0*neuron_spacing
    neurons = NeuronGroup(
        N,
        """
        g_m :siemens
        C_m :farad
        V_thr :volt
        V_reset :volt
        V_L :volt
        tau_GABA :second
        V_I :volt
        g_GABA :siemens
        s_NMDA_tot:1
        lamda_NA:1
        g_AMPA_rec:siemens
        g_NMDA:siemens
        Mg2 :1
        I_background:amp
        c1:amp
        c2:amp
        c3:1
        c4:amp
        c5:amp
        c6:amp
        c7:amp
        mu_e:1
        mu_i:1
        
        g_AMPA_ext: siemens
        dv / dt = (- g_m * (v - V_L) - I_syn) / C_m : volt (unless refractory)
        I_syn = I_AMPA_ext + I_AMPA_rec + I_NMDA_rec + I_GABA_rec+I_soma_dend +I_background: amp
        I_AMPA_ext = g_AMPA_ext * (v - 0. * mV) * s_AMPA_ext : amp
        ds_AMPA_ext / dt = - s_AMPA_ext / ( 2. * ms) : 1
        I_GABA_rec = mu_i*g_GABA * (v - V_I) * s_GABA : amp
        ds_GABA / dt = - s_GABA / tau_GABA : 1 
        I_AMPA_rec =mu_e*lamda_NA* g_AMPA_rec * (v - 0. * mV) *s_AMPA : amp
        ds_AMPA / dt = - s_AMPA / (2. * ms): 1 
        I_NMDA_rec = mu_e*lamda_NA*g_NMDA * (v - 0. * mV) / (1 + Mg2 * exp(-0.062 * v / mV) / 3.57) * s_NMDA_tot : amp
        
        I_dend_inh=I_GABA_rec:amp
        I_dend_exc=I_AMPA_ext + I_AMPA_rec + I_NMDA_rec +I_background :amp
        num=-I_dend_inh/c6 :1
        mid=(I_dend_exc +c3*I_dend_inh + c4)/c5*exp(num):1
        I_soma_dend=c1*tanh(mid) + c2 :amp

        y:metre
        
        """,
        threshold="v>V_thr",
        reset="v=V_reset",
        refractory=tau_rp,
        method='euler',
        name=name,
    )
    neurons.V_thr=V_thr
    neurons.V_reset = V_reset
    neurons.V_L = V_L
    neurons.g_m = g_m
    neurons.C_m = C_m
    neurons.g_AMPA_ext = g_AMPA_ext
    neurons.V_I = V_I
    neurons.tau_GABA = tau_GABA
    neurons.g_GABA = g_GABA
    neurons.g_AMPA_rec=g_AMPA_rec
    neurons.g_NMDA=g_NMDA
    neurons.lamda_NA=lamda_NA
    neurons.Mg2 = Mg2
    neurons.c1=c1
    neurons.c2=c2
    neurons.c3=c3
    neurons.c4=c4
    neurons.c5=c5
    neurons.c6=c6
    neurons.c7 = c7
    neurons.I_background=I_background
    neurons.mu_e=mu_e
    neurons.mu_i=mu_i
    neurons.y='i*neuron_spacing'


    return neurons




def get_synapses(name, source, target,weight, eqs,g_AMPA_rec,g_NMDA,alpha_ampa,alpha_nmda,tau_NMDA_decay,tau_facil=None):
    """Construct connections and retrieve synapses

    name -- name of synapses
    source -- source of connections
    target -- target of connections
    weight -- weight matrix

    """


    synapses_eqs = """
    tau_NMDA_decay :second
    U :1
    tau_u :second
    tau_D :second
    alpha_ampa:1
    alpha_nmda:1
    

    
    dx / dt = (1- x) / tau_D : 1 (clock-driven)
    du / dt=(U-u)/tau_u :1 (clock-driven)    
    
    ds_NMDA / dt = - s_NMDA / tau_NMDA_decay: 1 (clock-driven)
    
    
    
    
    """.format(
        name
    )

    synapses_eqs+=eqs

    if tau_facil:

        synapses_action = """
        x += -u*x
        u += U*(1-u) 
        s_AMPA += x*u*alpha_ampa
        s_NMDA+= x*u*(1- s_NMDA)*alpha_nmda
        """
    else:
        synapses_action = """
        s_GABA +=1
        """

    synapses = Synapses(
        source,
        target,
        model=synapses_eqs,
        on_pre=synapses_action,
        method='euler',
        name=name,
    )

    
    alpha = 0.5 / ms
    # alpha_ampa=2.5*5
    # alpha_nmda=2.5*5

    U = 0.15
    tau_u = 1500 * ms
    tau_D = 2 * ms

    synapses.connect()


    synapses.tau_NMDA_decay = tau_NMDA_decay


    synapses.U = U
    synapses.tau_u = tau_u
    synapses.tau_D = tau_D

    synapses.w=weight
    synapses.x=1
    synapses.u=0

    synapses.alpha_ampa=alpha_ampa
    synapses.alpha_nmda=alpha_nmda

    return synapses

def weight_balance(w):
    raw=w.shape[0]
    col=w.shape[1]

    for num in range(col):
        sum_excitation=np.sum(w[:,num])
        w[:, num]+=-sum_excitation/raw
    return w


def input(num,t,P_E,P_I,sub_E,sub_I):
    f = 0.1
    C_ext = 900
    C_selection = int(f * C_ext)
    rate_selection = 50 * Hz
    stimuli1 = TimedArray(np.r_[np.zeros(40), np.ones(20), np.zeros(400)], dt=25 * ms)
    input_e = PoissonInput(P_E[:sub_E*num], 's_AMPA_ext', C_selection, rate_selection, 'stimuli1(t)')
    input_i=PoissonInput(P_I[:sub_I*num], 's_AMPA_ext', C_selection, rate_selection, 'stimuli1(t)')

    stimuli_reset = TimedArray(np.r_[np.zeros(int(40*t)), np.ones(2), np.zeros(400)], dt=25 * ms)
    input_reset_I = PoissonInput(P_E[:sub_E*num], 's_AMPA_ext', C_selection, rate_selection, 'stimuli_reset(t)')
    input_reset_E = PoissonInput(P_I[:sub_I*num], 's_AMPA_ext', C_selection, rate_selection, 'stimuli_reset(t)')


    return input_e,input_i,input_reset_E,input_reset_I,stimuli1,stimuli_reset

def get_0_1_array(w,p):
    '''按照数组模板生成对应的 0-1 矩阵，默认rate=0.2'''
    raw=w.shape[0]
    col=w.shape[1]
    array=np.ones((raw,col))
    rate=1-p
    zeros_num = int(array.size * rate)#根据0的比率来得到 0的个数
    new_array = np.ones(array.size)#生成与原来模板相同的矩阵，全为1
    new_array[:zeros_num] = 0 #将一部分换为0
    np.random.shuffle(new_array)#将0和1的顺序打乱
    re_array = new_array.reshape(array.shape)#重新定义矩阵的维度，与模板相同
    out=w*re_array
    return out

#将neuron的输出方式转换为放电序列时间矩阵
def change_matrix(spike_train,bin,t_start,t_end,N):
    t_delay=t_end-t_start
    R=np.zeros((t_delay,N))
    print(spike_train.t)
    for k in range(spike_train.t.shape[0]):
        t_=int(spike_train.t[k]/ms)
        n_=spike_train.i[k]
        print(t_)
        R[t_][n_]=1
    P=[]
    for k in range(int(t_delay/bin)):
        R_=R[k*bin:(k+1)*bin]
        R_=R_.sum(0)
        P.append(R_)
    return P

#计算神经元的皮尔逊相关系数；计算拟合后各个神经元衰减曲线的时间尺度
def time_scale(P,bin,t_start,t_end,N):
    t_delay=t_end-t_start
    count=int(t_delay/bin)

    ac=[]
    t=[]
    s_=int(t_start/bin)
    e_=int(t_end/bin)
    P=P[s_:e_]
    count=0
    for k in P:
        ac_=pearsonr(P[0],k)
        ac.append(ac_[0]) #相关系数
        t.append(count) #间隔时间
        count=count+1
    def func(t, tau):
        return np.exp(-t / tau)
    ac=np.nan_to_num(ac)
    popt1, pcov1 = curve_fit(func, t, ac, maxfev=20000, method='dogbox')
    tau=popt1[0]
    return ac,t,tau


def stable_spatial(P):
    pca=PCA(n_components=2)
    pca.fit(P)
    P_pca=pca.transform(P)
    var=pca.explained_variance_ratio_
    return P_pca,var


def main_work():
    # populations
    N = 1000
    N_E = int(N * 0.8)  # pyramidal neurons
    N_I = int(N * 0.2)  # interneurons
    sub =4  # number of input pool
    sub_E = int(N_E / sub)  # neuron number in input pool
    sub_I=int(N_I / sub)

    # membrane capacitance
    C_m_E = 0.5 * nF
    C_m_I = 0.2 * nF

    # membrane leak
    g_m_E = 25. * nS
    g_m_I = 20. * nS

    # refractory period
    tau_rp_E = 2. * ms
    tau_rp_I = 1. * ms

    # external stimuli
    rate = 3 * Hz
    C_ext = 800


    # AMPA (excitatory)
    g_AMPA_ext_E = 2.08 * nS
    g_AMPA_rec_E = 0.104 * nS * 800. / N_E
    g_AMPA_ext_I = 1.62 * nS
    g_AMPA_rec_I = 0.081 * nS * 800. / N_E

    # NMDA (excitatory)
    g_NMDA_E = 0.327 * nS * 800. / N_E
    g_NMDA_I = 0.258 * nS * 800. / N_E

    # GABAergic (inhibitory)
    g_GABA_E = 1.25 * nS * 200. / N_I
    g_GABA_I = 0.973 * nS * 200. / N_I

    # subpopulations
    f = 0.1
    
    

    #background current
    I_background_e=310 *pA
    I_background_i = 300 * pA

    
    tau_GABA = 10. * ms
    tau_NMDA_decay = 100. * ms

    para=np.load('./ampa-nmda/ampa-nmda.npy')
    pr=[]
    pr.append(para[:,7])
    pr.append(para[:,10])
    nonzero_counts = np.count_nonzero(pr, axis=0)
    # 将数组转换为 pandas 的 Series 对象
    series = pd.Series(nonzero_counts)

    # 使用 value_counts 方法统计每个元素的数量
    value_counts = series.value_counts()

    # 获取零元素的数量
    zero_count = value_counts.get(0, 0)


    alpha_ampa_record=[]
    alpha_nmda_record=[]
    Wie_self_record=[]
    Wii_other_record=[]
    ac_e_pre_record=[]
    t_e_pre_record=[]
    tau_e_pre_record=[]
    ac_i_pre_record=[]
    t_i_pre_record=[]
    tau_i_pre_record=[]
    ac_e_post_record=[]
    t_e_post_recoed=[]
    tau_e_post_record=[]
    ac_i_post_record=[]
    t_i_post_recoed=[]
    tau_i_post_record=[]

    for one in para:
        #for two in range(4):
        
        remeber_num=4
        interval=0
        
        p_ii=0.3
        p_ie=0.8
        p_ee=0.3
        p_ei=0.8
        
        W_positive =6.217916
        W_negative =1.298013
        lamda_NA=1
        alpha_ampa=one[13]
        alpha_nmda=one[14]
        character=2


        P_E = get_population("P_E", N_E, tau_rp_E, g_m_E, C_m_E,g_AMPA_ext_E,g_GABA_E,g_AMPA_rec_E,g_NMDA_E,I_background_e,lamda_NA,tau_GABA)
        P_I = get_population("P_I", N_I, tau_rp_I, g_m_I, C_m_I,g_AMPA_ext_I,g_GABA_I,g_AMPA_rec_I,g_NMDA_I,I_background_i,lamda_NA,tau_GABA)

        

        
        
        # average field weight
        Wei_self=19.948639
        Wei_other=17.144662
        Wie_self=one[7]
        Wie_other=13.155368
        Wii_self=18.437482
        Wii_other=one[10]
        # Wei_self=10
        # Wei_other=2
        # Wie_self=0
        # Wie_other=1
        # Wii_self=2
        # Wii_other=0



        # compute weight
        Matrix_weight_E_E = np.zeros((N_E, N_E))
        Matrix_weight_E_I = np.zeros((N_E, N_I))
        Matrix_weight_I_E = np.zeros((N_I, N_E))
        Matrix_weight_I_I = np.zeros((N_I, N_I))

        Matrix_weight_E_I[:] = Wei_other
        Matrix_weight_I_E[:] = Wie_other
        Matrix_weight_I_I[:] = Wii_other


        for num in range(sub):
            Matrix_weight_E_E[num * sub_E:(num + 1) * sub_E, num * sub_E:(num + 1) * sub_E] = W_positive
            Matrix_weight_E_I[num * sub_E:(num + 1) * sub_E,num * sub_I:(num + 1) * sub_I]=Wei_self
            Matrix_weight_I_E[num * sub_I:(num + 1) * sub_I,num * sub_E:(num + 1) * sub_E]=Wie_self
            Matrix_weight_I_I[num * sub_I:(num + 1) * sub_I,num * sub_I:(num + 1) * sub_I]=Wii_self

            #Matrix_weight_E_E[num * sub_E:(num + 1) * sub_E, num * sub_E:(num + 1) * sub_E] = np.random.rand()
            if num + 2 > sub:
                break
            Matrix_weight_E_E[num * sub_E:(num + 1) * sub_E, (num + 1) * sub_E:(num + 2) * sub_E] = W_negative
            Matrix_weight_E_E[(num + 1) * sub_E:(num + 2) * sub_E, num * sub_E:(num + 1) * sub_E] = W_negative
        

        

        Matrix_weight_E_E=weight_balance(Matrix_weight_E_E)
        Matrix_weight_E_I = weight_balance(Matrix_weight_E_I)
        Matrix_weight_I_E = weight_balance(Matrix_weight_I_E)
        Matrix_weight_I_I=weight_balance(Matrix_weight_I_I)

        Matrix_weight_E_E=get_0_1_array(Matrix_weight_E_E,p_ee)
        Matrix_weight_E_I=get_0_1_array(Matrix_weight_E_I,p_ei)
        Matrix_weight_I_E=get_0_1_array(Matrix_weight_I_E,p_ie)
        Matrix_weight_I_I=get_0_1_array(Matrix_weight_I_I,p_ii)



        Matrix_weight_E_E2 = Matrix_weight_E_E.flatten()
        Matrix_weight_E_I2 = Matrix_weight_E_I.flatten()
        Matrix_weight_I_E2 = Matrix_weight_I_E.flatten()
        Matrix_weight_I_I2 = Matrix_weight_I_I.flatten()

        tau_rec = 810 * ms
        # E to E
        eqs='''
        s_NMDA_tot_post=w*s_NMDA :1(summed)
        
        w : 1  
        '''
        C_E_E = get_synapses(
            "C_E_E",
            P_E,
            P_E,
            Matrix_weight_E_E2,
            eqs,
            g_AMPA_rec_E,
            g_NMDA_E,
            alpha_ampa,
            alpha_nmda,
            tau_NMDA_decay,
            tau_rec,
        )

        # E to I
        eqs = '''
            s_NMDA_tot_post=w*s_NMDA :1(summed)
            w : 1  
            '''
        C_E_I = get_synapses(
            "C_E_I",
            P_E,
            P_I,
            Matrix_weight_E_I2,
            eqs,
            g_AMPA_rec_E,
            g_NMDA_E,
            alpha_ampa,
            alpha_nmda,
            tau_NMDA_decay,
            tau_rec,
        )

        # I to I
        neuron_spacing = 50*umetre
        ii=2
        width = N/4.0*neuron_spacing
        eqs = '''
            
            w : 1  
            '''
        C_I_I =C_I_E = get_synapses(
            "C_I_I",
            P_I,
            P_I,
            Matrix_weight_I_I2,
            eqs,
            g_AMPA_rec_I,
            g_NMDA_I,
            alpha_ampa,
            alpha_nmda,
            tau_NMDA_decay,
        )
        # C_I_I.w = ii
        # I to E
        eqs = '''
            
            w : 1  
            '''
        C_I_E = get_synapses(
            "C_I_E",
            P_I,
            P_E,
            Matrix_weight_I_E2,
            eqs,
            g_AMPA_rec_I,
            g_NMDA_I,
            alpha_ampa,
            alpha_nmda,
            tau_NMDA_decay,
        )


        # external noise
        C_P_E = PoissonInput(P_E, 's_AMPA_ext', C_ext, rate, '1')
        C_P_I = PoissonInput(P_I, 's_AMPA_ext', C_ext, rate, '1')

        #每个记忆在不同的时间触发
        
        #每个记忆在不同的时间触发
        
        
        f = 0.1
        C_ext = 800
        C_selection = int(f * C_ext)
        rate_selection = 25 * Hz
        

        if remeber_num>=1:
            stimuli1 = TimedArray(np.r_[np.zeros(40), np.ones(16), np.zeros(400)], dt=25 * ms)
            input_e_1 = PoissonInput(P_E[:sub_E*1], 's_AMPA_ext', C_selection, rate_selection, 'stimuli1(t)')
            input_i_1=PoissonInput(P_I[:sub_I*1], 's_AMPA_ext', C_selection, rate_selection, 'stimuli1(t)')
            # stimuli_reset_1 = TimedArray(np.r_[np.zeros(int(40*(remeber_num*interval+1+cue))), np.ones(2), np.zeros(400)], dt=25 * ms)
            # input_reset_I_1 = PoissonInput(P_E[:sub_E*1], 's_AMPA_ext', C_selection, rate_selection, 'stimuli_reset_1(t)')
            # input_reset_E_1 = PoissonInput(P_I[:sub_I*1], 's_AMPA_ext', C_selection, rate_selection, 'stimuli_reset_1(t)')
        
        if remeber_num>=2:
            stimuli2 = TimedArray(np.r_[np.zeros(int(40+40*interval)), np.ones(16), np.zeros(400)], dt=25 * ms)
            input_e_2 = PoissonInput(P_E[sub_E*1:sub_E*2], 's_AMPA_ext', C_selection, rate_selection, 'stimuli2(t)')
            input_i_2=PoissonInput(P_I[sub_I*1:sub_I*2], 's_AMPA_ext', C_selection, rate_selection, 'stimuli2(t)')
            # stimuli_reset_2 = TimedArray(np.r_[np.zeros(int(40*interval+40*(remeber_num*interval+1+cue))), np.ones(2), np.zeros(400)], dt=25 * ms)
            # input_reset_I_2 = PoissonInput(P_E[sub_E*1:sub_E*2], 's_AMPA_ext', C_selection, rate_selection, 'stimuli_reset_2(t)')
            # input_reset_E_2 = PoissonInput(P_I[sub_I*1:sub_I*2], 's_AMPA_ext', C_selection, rate_selection, 'stimuli_reset_2(t)')
            
        if remeber_num>=3:
            stimuli3 = TimedArray(np.r_[np.zeros(int(40+40*interval*2)), np.ones(16), np.zeros(400)], dt=25 * ms)
            input_e_3 = PoissonInput(P_E[sub_E*2:sub_E*3], 's_AMPA_ext', C_selection, rate_selection, 'stimuli3(t)')
            input_i_3=PoissonInput(P_I[sub_I*2:sub_I*3], 's_AMPA_ext', C_selection, rate_selection, 'stimuli3(t)')
            cue1=0.6*2+0.2+2.5
            cue2=0.6*2+0.2+4
            # stimuli_reset_3 = TimedArray(np.r_[np.zeros(int(40*interval*2+40*(remeber_num*interval+1+cue))), np.ones(2), np.zeros(400)], dt=25 * ms)
            # input_reset_I_3 = PoissonInput(P_E[sub_E*2:sub_E*3], 's_AMPA_ext', C_selection, rate_selection, 'stimuli_reset_3(t)')
            # input_reset_E_3 = PoissonInput(P_I[sub_I*2:sub_I*3], 's_AMPA_ext', C_selection, rate_selection, 'stimuli_reset_3(t)')
        
        if remeber_num>=4:
            stimuli4 = TimedArray(np.r_[np.zeros(int(40+40*interval*3)), np.ones(16), np.zeros(400)], dt=25 * ms)
            input_e_4 = PoissonInput(P_E[sub_E*3:sub_E*4], 's_AMPA_ext', C_selection, rate_selection, 'stimuli4(t)')
            input_i_4=PoissonInput(P_I[sub_I*3:sub_I*4], 's_AMPA_ext', C_selection, rate_selection, 'stimuli4(t)')
            # stimuli_reset_4 = TimedArray(np.r_[np.zeros(int(40*interval*3+40*(remeber_num*interval+1+cue))), np.ones(2), np.zeros(400)], dt=25 * ms)
            # input_reset_I_4 = PoissonInput(P_E[sub_E*3:sub_E*4], 's_AMPA_ext', C_selection, rate_selection, 'stimuli_reset_4(t)')
            # input_reset_E_4 = PoissonInput(P_I[sub_I*3:sub_I*4], 's_AMPA_ext', C_selection, rate_selection, 'stimuli_reset_4(t)')
        
        if remeber_num>=5:
            stimuli5 = TimedArray(np.r_[np.zeros(int(40+40*interval*4)), np.ones(16), np.zeros(400)], dt=25 * ms)
            input_e_5 = PoissonInput(P_E[sub_E*4:sub_E*5], 's_AMPA_ext', C_selection, rate_selection, 'stimuli5(t)')
            input_i_5=PoissonInput(P_I[sub_I*4:sub_I*5], 's_AMPA_ext', C_selection, rate_selection, 'stimuli5(t)')
            # stimuli_reset_5 = TimedArray(np.r_[np.zeros(int(40*interval*4+40*(remeber_num*interval+1+cue))), np.ones(2), np.zeros(400)], dt=25 * ms)
            # input_reset_I_5 = PoissonInput(P_E[sub_E*4:sub_E*5], 's_AMPA_ext', C_selection, rate_selection, 'stimuli_reset_5(t)')
            # input_reset_E_5 = PoissonInput(P_I[sub_I*4:sub_I*5], 's_AMPA_ext', C_selection, rate_selection, 'stimuli_reset_5(t)')
        
        if remeber_num>=6:
            stimuli6 = TimedArray(np.r_[np.zeros(int(40+40*interval*5)), np.ones(16), np.zeros(400)], dt=25 * ms)
            input_e_6 = PoissonInput(P_E[sub_E*5:sub_E*6], 's_AMPA_ext', C_selection, rate_selection, 'stimuli6(t)')
            input_i_6=PoissonInput(P_I[sub_I*5:sub_I*6], 's_AMPA_ext', C_selection, rate_selection, 'stimuli6(t)')
            # stimuli_reset_6 = TimedArray(np.r_[np.zeros(int(40*interval*5+40*(remeber_num*interval+1+cue))), np.ones(2), np.zeros(400)], dt=25 * ms)
            # input_reset_I_6 = PoissonInput(P_E[sub_E*5:sub_E*6], 's_AMPA_ext', C_selection, rate_selection, 'stimuli_reset_6(t)')
            # input_reset_E_6 = PoissonInput(P_I[sub_I*5:sub_I*6], 's_AMPA_ext', C_selection, rate_selection, 'stimuli_reset_6(t)')
            
        if remeber_num>=7:
            stimuli7 = TimedArray(np.r_[np.zeros(int(40+40*interval*6)), np.ones(16), np.zeros(400)], dt=25 * ms)
            input_e_7 = PoissonInput(P_E[sub_E*6:sub_E*7], 's_AMPA_ext', C_selection, rate_selection, 'stimuli7(t)')
            input_i_7=PoissonInput(P_I[sub_I*6:sub_I*7], 's_AMPA_ext', C_selection, rate_selection, 'stimuli7(t)')
            # stimuli_reset_7 = TimedArray(np.r_[np.zeros(int(40*interval*6+40*(remeber_num*interval+1+cue))), np.ones(2), np.zeros(400)], dt=25 * ms)
            # input_reset_I_7 = PoissonInput(P_E[sub_E*6:sub_E*7], 's_AMPA_ext', C_selection, rate_selection, 'stimuli_reset_7(t)')
            # input_reset_E_7 = PoissonInput(P_I[sub_I*6:sub_I*7], 's_AMPA_ext', C_selection, rate_selection, 'stimuli_reset_7(t)')
            
        if remeber_num>=8:
            stimuli8 = TimedArray(np.r_[np.zeros(int(40+40*interval*7)), np.ones(16), np.zeros(400)], dt=25 * ms)
            input_e_8 = PoissonInput(P_E[sub_E*7:sub_E*8], 's_AMPA_ext', C_selection, rate_selection, 'stimuli8(t)')
            input_i_8=PoissonInput(P_I[sub_I*7:sub_I*8], 's_AMPA_ext', C_selection, rate_selection, 'stimuli8(t)')
            # stimuli_reset_8 = TimedArray(np.r_[np.zeros(int(40*interval*7+40*(remeber_num*interval+1+cue))), np.ones(2), np.zeros(400)], dt=25 * ms)
            # input_reset_I_8 = PoissonInput(P_E[sub_E*7:sub_E*8], 's_AMPA_ext', C_selection, rate_selection, 'stimuli_reset_8(t)')
            # input_reset_E_8 = PoissonInput(P_I[sub_I*7:sub_I*8], 's_AMPA_ext', C_selection, rate_selection, 'stimuli_reset_8(t)')
                
        if remeber_num>=9:
            stimuli9 = TimedArray(np.r_[np.zeros(int(40+40*interval*8)), np.ones(16), np.zeros(400)], dt=25 * ms)
            input_e_9 = PoissonInput(P_E[sub_E*8:sub_E*9], 's_AMPA_ext', C_selection, rate_selection, 'stimuli9(t)')
            input_i_9=PoissonInput(P_I[sub_I*8:sub_I*9], 's_AMPA_ext', C_selection, rate_selection, 'stimuli9(t)')
            # stimuli_reset_9 = TimedArray(np.r_[np.zeros(int(40*interval*8+40*(remeber_num*interval+1+cue))), np.ones(2), np.zeros(400)], dt=25 * ms)
            # input_reset_I_9 = PoissonInput(P_E[sub_E*8:sub_E*9], 's_AMPA_ext', C_selection, rate_selection, 'stimuli_reset_9(t)')
            # input_reset_E_9 = PoissonInput(P_I[sub_I*8:sub_I*9], 's_AMPA_ext', C_selection, rate_selection, 'stimuli_reset_9(t)')



        cue1=1.4+1
        cue2=0.6*(remeber_num-1)+0.2+4+1
        sim_cue=cue1
        stimuli_reset = TimedArray(np.r_[np.zeros(int(40*sim_cue)), np.ones(2), np.zeros(400)], dt=25 * ms)
        input_reset_I = PoissonInput(P_E[:N_E], 's_AMPA_ext', C_selection, rate_selection, 'stimuli_reset(t)')
        input_reset_E = PoissonInput(P_I[:N_I], 's_AMPA_ext', C_selection, rate_selection, 'stimuli_reset(t)')
        
        #input_e,input_i,input_reset_E,input_reset_I,stimuli1,stimuli_reset=input(remeber_num,cue,P_E,P_I,sub_E,sub_I)

        e_neuron = [SpikeMonitor(P_E[pi:pi + sub_E]) for pi in range(0, 0 + 4* sub_E, sub_E)]
        i_neuron = [SpikeMonitor(P_I[pi:pi + sub_I]) for pi in range(0, 0 + 4* sub_I, sub_I)]
        e_neuron_all=SpikeMonitor(P_E)
        i_neuron_all=SpikeMonitor(P_I)

        r_E_sels = [PopulationRateMonitor(P_E[pi:pi + sub_E]) for pi in range(0, 0 + 4* sub_E, sub_E)]
        r_I_sels = [PopulationRateMonitor(P_I[pi:pi + sub_I]) for pi in range(0, 0 + 4* sub_I, sub_I)]
        r_E = PopulationRateMonitor(P_E)
        r_I = PopulationRateMonitor(P_I)
        net = Network(collect())
        net.active=True
        net.add(e_neuron)
        net.add(i_neuron)
        net.run(5* second, report='stdout')


        bin=50 #ms
        #转化神经元为放电数矩阵;行是时间。列是N
        t_start=0
        t_end=5000
        E_count=change_matrix(e_neuron_all,bin,t_start,t_end,N_E)
        I_count=change_matrix(i_neuron_all,bin,t_start,t_end,N_I)

        
        #时间常数计算
        t_start=1000
        t_end=2400
        ac_e_pre,t_e_pre,tau_e_pre=time_scale(E_count,bin,t_start,t_end,N_E)
        ac_i_pre,t_i_pre,tau_i_pre=time_scale(I_count,bin,t_start,t_end,N_I)
        t_start=2400
        t_end=4500
        ac_e_post,t_e_post,tau_e_post=time_scale(E_count,bin,t_start,t_end,N_E)
        ac_i_post,t_i_post,tau_i_post=time_scale(I_count,bin,t_start,t_end,N_I)
        
        alpha_ampa_record.append(alpha_ampa)
        alpha_nmda_record.append(alpha_nmda)
        Wie_self_record.append(Wie_self)
        Wii_other_record.append(Wii_other)
        ac_e_pre_record.append(ac_e_pre)
        t_e_pre_record.append(t_e_pre)
        tau_e_pre_record.append(tau_e_pre)
        ac_i_pre_record.append(ac_i_pre)
        t_i_pre_record.append(t_i_pre)
        tau_i_pre_record.append(tau_i_pre)
        ac_e_post_record.append(ac_e_post)
        t_e_post_recoed.append(t_e_post)
        tau_e_post_record.append(tau_e_post)
        ac_i_post_record.append(ac_i_post)
        t_i_post_recoed.append(t_i_post)
        tau_i_post_record.append(tau_i_post)
        

        net.active=False
        net.stop()

        if remeber_num>=1:
            del stimuli1
            del input_e_1
            del input_i_1
            # del input_reset_I_1
            # del input_reset_E_1
            
        
        if remeber_num>=2:
            del stimuli2
            del input_e_2 
            del input_i_2
            # del input_reset_I_2
            # del input_reset_E_2
            
            
        if remeber_num>=3:
            del stimuli3
            del input_e_3 
            del input_i_3
            # del input_reset_I_3
            # del input_reset_E_3
            
        if remeber_num>=4:
            del stimuli4
            del input_e_4 
            del input_i_4
            # del input_reset_I_4
            # del input_reset_E_4
            
        

        del net
        del P_E
        del P_I
        del input_reset_I
        del input_reset_E
    array_dict={'alpha_ampa_record':alpha_ampa_record,
        'alpha_nmda_record':alpha_nmda_record,
        'Wie_self_record':Wie_self_record,
        'Wii_other_record':Wii_other_record,
        'ac_e_pre_record':ac_e_pre_record,
        't_e_pre_record':t_e_pre_record,
        'tau_e_pre_record':tau_e_pre_record,
        'ac_i_pre_record':ac_i_pre_record,
        't_i_pre_record':t_i_pre_record,
        'tau_i_pre_record':tau_i_pre_record,
        'ac_e_post_record':ac_e_post_record,
        't_e_post_recoed':t_e_post_recoed,
        'tau_e_post_record':tau_e_post_record,
        'ac_i_post_record':ac_i_post_record,
        't_i_post_recoed':t_i_post_recoed,
        'tau_i_post_record':tau_i_post_record}

    np.savez('ampa-nmda-out.npz',**array_dict)


            




    
main_work()
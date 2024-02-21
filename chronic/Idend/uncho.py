from brian2 import *
import numpy as np
import matplotlib.pyplot as plt
import random
from scipy.stats import pearsonr
from scipy.optimize import curve_fit
from sklearn.decomposition import PCA

import re
plt.switch_backend('agg')

regx=re.compile(r'\d+.\d+')


def get_population(name,N,tau_rp,g_m,C_m,g_AMPA_ext,g_GABA,g_AMPA_rec,g_NMDA,I_background,lamda_NA,tau_GABA,ampa_scale,nmda_scale,mu_ampa,mu_nmda,mu_gaba,gaba,dend_inh):
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
        ampa_scale:1
        nmda_scale:1
        mu_ampa:1
        mu_nmda:1
        mu_gaba:1
        gaba :1
        dend_inh :1
        

        
        g_AMPA_ext: siemens
        dv / dt = (- g_m * (v - V_L) - I_syn) / C_m : volt (unless refractory)
        I_syn = I_AMPA_ext + I_AMPA_rec*ampa_scale + I_NMDA_rec*nmda_scale + gaba*I_GABA_rec+I_soma_dend +I_background: amp
        I_AMPA_ext = g_AMPA_ext * (v - 0. * mV) * s_AMPA_ext : amp
        ds_AMPA_ext / dt = - s_AMPA_ext / ( 2. * ms) : 1
        I_GABA_rec =mu_gaba*g_GABA * (v - V_I) * s_GABA : amp
        ds_GABA / dt = - s_GABA / tau_GABA : 1 
        I_AMPA_rec =mu_ampa*lamda_NA* g_AMPA_rec * (v - 0. * mV) *s_AMPA : amp
        ds_AMPA / dt = - s_AMPA / (2. * ms): 1 
        I_NMDA_rec = mu_nmda*lamda_NA*g_NMDA * (v - 0. * mV) / (1 + Mg2 * exp(-0.062 * v / mV) / 3.57) * s_NMDA_tot : amp
        
        I_dend_inh=I_GABA_rec:amp
        I_dend_exc=I_AMPA_ext + I_AMPA_rec + I_NMDA_rec +I_background :amp
        num=-I_dend_inh/c6*dend_inh :1
        mid=(I_dend_exc +c3*I_dend_inh + c4)/c5*exp(num):1
        I_soma_dend=c1*tanh(mid) + c2 :amp

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
    neurons.ampa_scale=ampa_scale
    neurons.nmda_scale=nmda_scale
    neurons.mu_ampa=mu_ampa
    neurons.mu_nmda=mu_nmda
    neurons.mu_gaba=mu_gaba
    neurons.gaba=gaba
    neurons.dend_inh=dend_inh




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

def rand_seq(mean,std,n):
    x=[]
    for i in range(n):
        a=np.random.normal(loc=mean, scale=std, size=None)
        x.append(a)
    return x


def main_work():
    # populations
    N = 1000
    N_E = int(N * (47/57.4)/4)*4 # pyramidal neurons 2/3
    N_I =168 #int(N * ((57.4-47)/57.4)/3)*3  # interneurons both devide by3and 4
    sub =4  # number of input pool
    sub_E = int(N_E / sub)  # neuron number in input pool
    sub_I=int(N_I / sub)

    # membrane capacitance
    C_m_E =rand_seq(164.96,59.11,N_E) * nF
    fs=rand_seq( 59.58,10.59,int(N_I/3))
    bt=rand_seq(79.36,14.83,int(N_I/3))
    mc=rand_seq(81.12,28.96,int(N_I/3))
    c_i=fs+bt+mc
    C_m_I = c_i * nF

    # membrane leak
    g_m_E = rand_seq(7.04,1.72,N_E)* nS
    fs=rand_seq( 5.34,0.91,int(N_I/3))
    bt=rand_seq(3.99,0.51,int(N_I/3))
    mc=rand_seq( 2.98,0.55,int(N_I/3))
    g_i=fs+bt+mc
    g_m_I = g_i * nS

    # refractory period
    tau_rp_E = 2. * ms
    tau_rp_I = 1. * ms

    # external stimuli
    rate = 3 * Hz
    C_ext = 800


    # AMPA (excitatory)
    g_AMPA_ext_E = 2.08 * nS
    g_AMPA_rec_E = 0.104 * nS 
    g_AMPA_ext_I = 1.62 * nS
    g_AMPA_rec_I = 0.081 * nS 

    # NMDA (excitatory)
    g_NMDA_E = 0.327 * nS 
    g_NMDA_I = 0.258 * nS 

    # GABAergic (inhibitory)
    g_GABA_E = 1.25 * nS 
    g_GABA_I = 0.973 * nS 

    # subpopulations
    f = 0.1
    
    

    #background current
    I_background_e=310 *pA
    I_background_i = 300 * pA

    
    tau_GABA = 10. * ms
    tau_NMDA_decay = 100. * ms
    alph_ampa=[]
    alph_nmda=[]

    nm_pre_swi=197
    nm_pos_swi=425
    alph_nmda.append(nm_pos_swi/nm_pre_swi)
    am_pre_swi=58.6
    am_pos_swi=98.8
    alph_ampa.append(am_pos_swi/am_pre_swi)

    nm_pre_yue=127
    nm_pos_yue=319
    alph_nmda.append(nm_pos_yue/nm_pre_yue)
    am_pre_yue=52.5
    am_pos_yue=115
    alph_ampa.append(am_pos_yue/am_pre_yue)

    nm_pre_ping=154.5
    nm_pos_ping=385.6
    alph_nmda.append(nm_pos_ping/nm_pre_ping)
    am_pre_ping=53.4
    am_pos_ping=99
    alph_ampa.append(am_pos_ping/am_pre_ping)

    nm_pre_piz=168
    nm_pos_piz=361
    alph_nmda.append(nm_pos_piz/nm_pre_piz)
    am_pre_piz=65
    am_pos_piz=141
    alph_ampa.append(am_pos_piz/am_pre_piz)




    # para=np.load('./ampa-nmda/ampa-nmda.npy')
    alpha_ampa_record=[]
    alpha_nmda_record=[]
    Wie_self_record=[]
    Wii_other_record=[]
    ampa_scale_record=[]
    nmda_scale_record=[]
    e_spike=[]
    t_e_spike=[]
    i_spike=[]
    t_i_spike=[]

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
    para_record=[]

    para=np.load('./i-inh-dend/dend-inh.npy')



    for one in para:
        #for two in range(4):
        
        remeber_num=4
        interval=0
        
        p_ii=0.250
        p_ie=(0.466+0.301+0.710)/3
        p_ee=0.139
        p_ei=( 0.325+0.159+0.290)/3

        W_positive =one[12]
        W_negative =one[13]
        
        while W_negative>W_positive:
            W_positive =random.uniform(0, 10)
            W_negative =random.uniform(0, 10)
        lamda_NA=1

        # ampa_scale=alph_ampa[two]
        # nmda_scale=alph_nmda[two]
        ampa_scale=1
        nmda_scale=1

        fac=0

        BP=[50,101,366]
        MP=[582,667,174,121,215,291,22]
        BP=np.array(BP)
        MP=np.array(MP)
        EXN=164*(1-fac)
        MP_IN=7.6*0.01
        BP_IN=26.2*0.01
        EXN_IN=9.3*0.01

        mu_ampa_e=EXN*(1-EXN_IN)*0.5
        mu_nmda_e=EXN*(1-EXN_IN)*0.5
        mu_gaba_e=EXN*EXN_IN

        mu_ampa_i=np.mean(BP)*(1-BP_IN)*0.5+np.mean(MP)*(1-MP_IN)*0.5
        mu_nmda_i=np.mean(BP)*(1-BP_IN)*0.5+np.mean(MP)*(1-MP_IN)*0.5
        mu_gaba_i=np.mean(BP)*(BP_IN)+np.mean(MP)*(MP_IN)


        alpha_ampa=one[14]
        alpha_nmda=one[15]


        # #I_gaba charactor
        # gaba_e=1
        # gaba_i=1
        # #I_dend_inh charactor
        # dend_inh_e=1
        # dend_inh_i=1

        
        gaba_e=one[0]
        gaba_i=1
        #I_dend_inh charactor
        dend_inh_e=one[1]
        dend_inh_i=1



        P_E = get_population("P_E", N_E, tau_rp_E, g_m_E, C_m_E,g_AMPA_ext_E,g_GABA_E,g_AMPA_rec_E,g_NMDA_E,I_background_e,lamda_NA,tau_GABA,ampa_scale,nmda_scale,mu_ampa_e,mu_nmda_e,mu_gaba_e,gaba_e,dend_inh_e)
        P_I = get_population("P_I", N_I, tau_rp_I, g_m_I, C_m_I,g_AMPA_ext_I,g_GABA_I,g_AMPA_rec_I,g_NMDA_I,I_background_i,lamda_NA,tau_GABA,ampa_scale,nmda_scale,mu_ampa_i,mu_nmda_i,mu_gaba_i,gaba_i,dend_inh_i)

        

        
        
        # average field weight

        Wei_self = one[6]
        Wei_other = one[7]
        Wie_self = one[8]
        Wie_other = one[9]
        Wii_self = one[10]
        Wii_other = one[11]




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
        

        
        stimuli = TimedArray(np.r_[np.zeros(int(40)), np.ones(32), np.zeros(400)], dt=25 * ms)
        input_e = PoissonInput(P_E, 's_AMPA_ext', C_selection, rate_selection, 'stimuli(t)')
        input_i=PoissonInput(P_I, 's_AMPA_ext', C_selection, rate_selection, 'stimuli(t)')
            
        


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

        e_neuron = np.array(e_neuron_all.i)
        i_neuron = np.array(i_neuron_all.i)
        t_e = np.array(e_neuron_all.t)
        t_i = np.array(i_neuron_all.t)

        e_ = 200
        i_ = 40
        i_index = []
        i_t_index = []
        e_index = []
        e_t_index = []

        for k in range(i_neuron.shape[0]):
            group = int(i_neuron[k] / sub_I)
            if (i_neuron[k] < i_ + group * sub_I) and (i_neuron[k] > group * sub_I):
                ind = i_neuron[k] - sub_I * group
                index_ = ind + group * (e_ + i_) + e_
                i_index.append(index_)
                i_t_index.append(t_i[k])

        for k in range(e_neuron.shape[0]):
            group = int(e_neuron[k] / sub_E)
            if (e_neuron[k] < e_ + group * sub_E) and (e_neuron[k] > group * sub_E):
                ind = e_neuron[k] - sub_E * group
                index_ = ind + group * (e_ + i_)
                e_index.append(index_)
                e_t_index.append(t_e[k])
        size=5
        
        
        plt.scatter(i_t_index,i_index,c="#676FA3",s=size)
        plt.scatter(e_t_index,e_index,c="#FF5959",s=size)
        plt.xlabel("X")
        plt.ylabel("Y")
        plt.title("Scatter Plot of two different datasets")
        plt.savefig("./no-cho/dend/dend%f.jpg"%(one[1]))
        plt.close()
        array_dict={'e_neuron':e_neuron,
                    'i_neuron':i_neuron,
                    't_e':t_e,
                    't_i':t_i}

        np.savez('./no-cho/dend/dend%f.npz'%(one[1]),**array_dict)

            
        

        net.active=False
        net.stop()

        
        del stimuli
        del input_e
        del input_i
            
        

        del net
        del P_E
        del P_I
        del input_reset_I
        del input_reset_E
    

            




    
main_work()
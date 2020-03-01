from scipy import signal
import numpy as np
import matplotlib.pyplot as plt
import pandas
from scipy.stats import linregress

def Sedlak_CF(volt, R = 4):
    '''
    Maps an N*1 array of voltage readings to the characteristic wave via the
    characteristic function put forth by Sedlak et. al. 2013

    volt: array of voltage readings corresponding to time (N*1 array-like)
    R: parameter of the characteristic function

    returns:
    CF_volt (array-like)
    '''
    CF_volt = np.ones(len(volt))*abs(volt[0])
    for i in range(1,len(volt)):
        CF_volt[i] = abs(volt[i])+R*abs(volt[i]-volt[i-1])
    return CF_volt

def squared_CF(volt):
    '''
    Maps an N*1 array of voltage readings to the characteristic wave via the
    characteristic function x(i) -> x(i)^2

    volt: array of voltage readings corresponding to time (N*1 array-like)

    returns:
    CF_volt (float)
    '''
    CF_volt = np.square(volt)
    return CF_volt

def abs_CF(volt):
    '''
    Maps an N*1 array of voltage readings to the characteristic wave via the
    characteristic function x(i) -> x(i)^2

    volt: array of voltage readings corresponding to time (N*1 array-like)

    returns:
    CF_volt (float)
    '''
    CF_volt = abs(volt)
    return CF_volt



def AIC(CF_volt, k):
    '''
    AIC value of signal through point k, assuming points are labeled from 1

    CF_volt: array of the characteristic functin of voltage readings
    corresponding to time (N*1 array-like)
    k: range of values for which AIC is calculated

    returns:
    AIC (float)
    '''
    N = len(CF_volt)
    if k <= 1:
        raise ValueError('k must be greater than 1')
    if k > N-2:
        raise ValueError('k must be less than len(CF_volt)-2')
    AIC = k*np.log(np.var(CF_volt[0:k]))+(N-k-1)*np.log(np.var(CF_volt[k: N]))
    return AIC





def get_arrival(CF_volt, time, delta= .1, tam=20, tfa=10, tfb=20):
    '''
    Gets arrival time of a single signal
    volt: array of voltage readings corresponding to time (N*1 array-like)
    time: array of times (N*1 array-like)
    delta: spacing between readings in microseconds
    tam = window parameter 1 in microseconds
    tfa = window parameter 2 in microseconds
    tfb = window parameter 3 in microseconds

    returns:
    arrival (float)
    '''
    k1 = np.argmax(CF_volt)
    tam = int(tam/delta) #number of indicies
    N = (k1+1)+tam    #length of signal to inspect

    if N > len(CF_volt-2):
        raise ValueError('tam window is too large, need to reduce')

    AIC1 = np.zeros(N)
    for k in range(2, N+1):
        AIC1[k-1] = AIC(CF_volt, k)
    AIC1[0]=AIC1[1]


    k2 = np.argmin(AIC1)
    tfb = int(tfb/delta)
    tfa = int(tfa/delta)

    cut = CF_volt[k2-tfb:]
    N2 = tfa+tfb

    AIC2 = np.zeros(N2)
    for k in range(2, N2+1):
        AIC2[k-1] = AIC(cut, k)
    AIC2[0]=AIC2[1]
    arrival = k2-tfb+np.argmin(AIC2) #index of arrival time

    return(time[arrival])

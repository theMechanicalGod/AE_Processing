import numpy as np
from scipy import signal
from ae_measure2 import *

def Sedlak_CF(volt, R = 4):
    '''
    Maps an N*1 array of voltage readings to the characteristic wave via the
    characteristic function put forth by Sedlak et. al. 2013
    volt: array of voltage readings corresponding to time (N*1 array-like)
    R: parameter of the characteristic function
    returns: CF_volt (array-like)
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
    CF_volt = np.abs(volt)
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


    if CF_volt[0]==CF_volt[1]: # cleans data s.t. AIC function works
        CF_volt[1]+=CF_volt[1]*.01

    AIC = k*np.log(np.var(CF_volt[0:k]))+(N-k-1)*np.log(np.var(CF_volt[k: N]))
    return AIC


def get_arrival(CF_volt, time=[], delta= .1, tam=20, tfa=10, tfb=20, out='index'):
    '''
    Gets arrival time of a single signal. Currently uses 1 pass of AIC window
    volt: array of voltage readings corresponding to time (N*1 array-like)
    time: array of times (N*1 array-like)                functionality to be added later
    delta: spacing between sensor readings in microseconds
    tam = window parameter 1 in microseconds
    tfa = window parameter 2 in microseconds
    tfb = window parameter 3 in microseconds
    out = determines if output is a time (float) or index (int)

    returns:
    arrival index (int)
    arrival time (float)
    '''



    k1 = np.argmax(CF_volt)
    tam = int(tam/delta) #number of indicies
    N = (k1+1)+tam    #length of signal to inspect

    if N > len(CF_volt)-2:
        #raise ValueError('tam window is too large, need to reduce')
        return(np.nan)

    AIC1 = np.zeros(N)
    cut1 = CF_volt[0:N]
    for k in range(2, N-2):
        AIC1[k-1] = AIC(cut1, k)
    AIC1[0]=AIC1[1]


    k2 = np.argmin(AIC1)
    '''
    # seems to work better with 1 iteration
    tfb = int(tfb/delta)
    tfa = int(tfa/delta)
    if k2+tfa+1>len(CF_volt):
        #print(k2, tfa)
        raise ValueError('something when wrong here')
    if k2-tfb<0:
        cut2 = CF_volt[0:k2+tfa]
    else:
        cut2 = CF_volt[k2-tfb:k2+tfa]
    N2 = len(cut2)
    AIC2 = np.zeros(N2)
    #print(cut2)
    for k in range(2, N2-2):
        AIC2[k-1] = AIC(cut2, k)
    AIC2[0]=AIC2[1]
    arrival = k2-tfb+np.argmin(AIC2) #index of arrival time
    '''

    if out=='time':
        return(time[k2])
    elif out=='index':
        return(k2)
    else:
        raise ValueError('Unexpected argument in out')



def get_first_peak_aic(wave, CF='Sedlak', N = 3):
    '''
    Finds first peak in an AE signal
    wave: signal, single event (array-like)
    CF: Characteristic function of AIC method
    N: Size of moving average

    returns:
    snipped: waveform with no pre-trigger noise (N*1 array-like)
    '''
    if CF == 'Sedlak':
        i = get_arrival(wave, out ='index')
        wave_no_noise = wave[i:]
        wave_smooth = np.abs(signal.convolve(wave_no_noise, np.ones(N)/N))
        peaks = signal.find_peaks(wave_smooth,height = 1e-6)

        return peaks[0][0]+i-1

def front_snip(signal, CF = 'Sedlak'):
    '''
    Snipping routine which removes pre-trigger noise
    signal: signal, single event (array-like)
    CF: Characteristic function of AIC method

    returns:
    snipped: waveform with no pre-trigger noise (N*1 array-like)
    '''
    if is_clipped(signal):
        raise ValueError('signal is clipped')
        return np.nan

    if CF == 'Sedlak':
        cf = Sedlak_CF(signal)

    index = get_arrival(cf)
    return signal[index:]



def end_snip(signal, delta= .1, window_size = 10, slide_length=40):
    '''
    Snipping routine which removes flexural wave component
    signal: signal, single event (array-like)
    delta: spacing between sensor readings in microseconds
    window_size: length of sliding window in microseconds
    slide_length: Number of microseconds the window slides over

    returns:
    snipped: waveform with no flexural component (N*1 array-like)
    '''
    window = window_size/delta
    slide_length = int(slide_length/delta)
    variance=[]

    cf = abs_CF(signal)
    maxIndex = np.argmax(np.abs(cf))

    if maxIndex-window/2 < 0:
        lower = 0
        upper = int(slide_length)
    elif maxIndex+window/2 > len(signal):
        raise ValueError('Poor waveform, inspect visually')
        return np.nan
    elif maxIndex+window/2+slide_length > len(signal):
        raise ValueError('Reduce slide length')
        return np.nan
    else:
        lower = int(maxIndex-window/2)
        upper = int(maxIndex+window/2)

    for i in range(slide_length):
        variance.append(np.var(cf[lower+i:upper+i]))
    index = maxIndex+np.argmin(variance)-lower
    return signal[:index]

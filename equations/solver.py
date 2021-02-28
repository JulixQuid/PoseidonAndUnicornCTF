import pandas as pd
import numpy as np
from scipy import optimize


# DL = cos–1 [{sinI1 × sinI2 × cos(A2 – A1)} + {cosI1 × cosI2}]
# RF = Tan(DL/2) × (180/π) × (2/DL)
# Δ N/S = [(sinI1 × cosA1) + (sinI2 × cosA2)] [R.F. × (ΔMD/2)]
# Δ E/W = [(sinI1 × sinA1) + (sinI2 × sinA2)] [R.F. × (ΔMD/2)]

class Solver:
    def __init__(self):
        self.data = pd.read_csv('data.csv')
        self.preprocesing()

    def preprocesing(self):
        self.data['inclination'] = np.radians(self.data['inclination'])
        self.data['delta_tmd'] = -self.data['tmd'].shift(1) + self.data['tmd']
        self.data['azimuth'] = np.radians(self.data['azimuth'])
        self.data['inclination_prev'] = self.data['inclination'].shift(1)
        self.data['azimuth_prev'] = self.data['azimuth'].shift(1)
        self.data = self.data.fillna(0)
        self.data[['calc_azi', 'calc_inc']] = self.data.apply(solve2, axis=1,
                                                              result_type="expand")
        self.data['calc_azi'] = self.data['calc_azi']%2*np.pi
        self.data['calc_inc'] = self.data['calc_inc']%np.pi
        self.data.to_csv('output.csv')


dns, dew, dmd, dtvd, dtmd, I1, A1 = 0,0,0,0,0,0,0
historic_i=[0]*3
historic_a=[0]*3

def f(x):
    A2, I2 = x
    global dns,dew,dmd, I1, A1
    return [(((np.sin(I1) * np.cos(A1)) + (np.sin(I2) * np.cos(A2)))*(dmd/2) - dns),
            (((np.sin(I1) * np.sin(A1)) + (np.sin(I2) * np.sin(A2))) *(dmd/2)- dew)]
def fns(x):
    A2, I2 = x
    global dns, dew, dmd, I1, A1
    return [(((np.sin(I1) * np.cos(A1)) + (np.sin(I2) * np.cos(A2))) * (dmd / 2) - dns) ,0]
def jacns(x):
    A2,I2 = x
    global dmd
    return np.array([[(0.5*dmd)*(np.sin(I2) * -np.sin(A2)), (0.5*dmd)*(np.cos(I2) * np.cos(A2))],[0,0]])

def few(x):
    A2, I2 = x
    global dns, dew, dmd, I1, A1
    return [(((np.sin(I1) * np.sin(A1)) + (np.sin(I2) * np.sin(A2))) * (dmd / 2) - dew) ,0]

def jacew(x):
    A2,I2 = x
    global dmd
    return np.array([[(0.5*dmd)*(np.sin(I2) * np.cos(A2)), (0.5*dmd)*(np.cos(I2) * np.sin(A2))],[0,0]])


def jacobian(x):
    A2,I2 = x
    global dmd
    return np.array([[(0.5*dmd)*(np.sin(I2) * -np.sin(A2)), (0.5*dmd)*(np.cos(I2) * np.cos(A2))],
                    [(0.5*dmd)*(np.sin(I2) * np.cos(A2)), (0.5*dmd)*(np.cos(I2) * np.sin(A2))]])

def solve(row):
    # Δ N/S = [(sinI1 * cosA1) + (sinI2 * cosA2)]
    # Δ E/W = [(sinI1 * sinA1) + (sinI2 * sinA2)]
    # find the root of
    #  f1(A2,I2) =  ([(sinI1 * cosA1) + (sinI2 * cosA2)] - Δ N/S)* (ΔMD/2)
    #  f2(A2,I2) =  ([(sinI1 * sinA1) + (sinI2 * sinA2)] - Δ E/W)* (ΔMD/2)
    # jacobian
    #  df1/dA2 = (ΔMD/2)*(sinI2 * -sinA2)
    #  df1/dI2 = (ΔMD/2)*(cosI2 * cosA2)
    #  df2/dA2 = (ΔMD/2)*(sinI2 * cosA2)
    #  df2/dI2 = (ΔMD/2)*(cosI2 * sinA2)
    global dns,dew,I1,A1,dmd,historic_a,historic_i
    dns = row['delta_ns']
    dew = row['delta_ew']
    dmd = row['delta_md']
    I1 = historic_i[-1]
    A1 = historic_a[-1]
    A2 = A1
    I2 = I1
    sol = optimize.root(f, [A2, I2], jac=jacobian, method='hybr')
    historic_i.append(list(sol.x)[1])
    historic_a.append(list(sol.x)[0])
    return list(sol.x)

def solve2(row):
    # Δ N/S = [(sinI1 * cosA1) + (sinI2 * cosA2)]
    # Δ E/W = [(sinI1 * sinA1) + (sinI2 * sinA2)]
    # find the root of
    #  f1(A2,I2) =  ([(sinI1 * cosA1) + (sinI2 * cosA2)] - Δ N/S)* (ΔMD/2)
    #  f2(A2,I2) =  ([(sinI1 * sinA1) + (sinI2 * sinA2)] - Δ E/W)* (ΔMD/2)
    # jacobian
    #  df1/dA2 = (ΔMD/2)*(sinI2 * -sinA2)
    #  df1/dI2 = (ΔMD/2)*(cosI2 * cosA2)
    #  df2/dA2 = (ΔMD/2)*(sinI2 * cosA2)
    #  df2/dI2 = (ΔMD/2)*(cosI2 * sinA2)
    global dns,dew,I1,A1,dmd,historic_a,historic_i
    dns = row['delta_ns']
    dew = row['delta_ew']
    dmd = row['delta_md']
    I1 = historic_i[-1]
    A1 = historic_a[-1]
    A2 = 0
    I2 = 0
    j=1000
    while j:
        sol = optimize.root(f, [A2, I2], jac=jacobian, method='hybr')
        if sol.success:
            final = list(sol.x)
            historic_i.append(list(sol.x)[1])
            historic_a.append(list(sol.x)[0])
            break
        A2 = np.random.random()*np.pi
        I2 = np.random.random()*np.pi
        j-=1
    else:
        final = [np.nan,np.nan]#list(sol.x)
    return final



def solve3(row):
    #Δ TVD = [cosI1 + cosI2] [R.F. × (Δ TMD/2)]
    global dns, dew, I1, A1, dmd, dtvd, dtmd, historic_a, historic_i
    I1 = historic_i[-1]
    I2 = np.mean(historic_a[-3:])
    j = 1000
    dtmd = row['delta_tmd']
    try:
        sol = optimize.root(f3, [I2], jac=False, method='hybr')
        if sol.success:
            final = list(sol.x)
            historic_i.append(list(sol.x)[0])
            #break
        I2 = np.random.random() * np.pi
        j -= 1
    except:
        print(I1)
    return final

def f3(x):
    # f3  = [cosI1 + cosI2] [R.F. × (Δ TMD/2)] - Δ TVD
    global I1,dtmd,dtvd
    I2 = x[0]
    return np.array([((np.cos(I1)+np.cos(I2))*(dtmd/2))-dtvd, ((np.cos(I1)+np.cos(I2))*(dtmd/2))-dtvd])

def jac3(x):
    I2 = x[0]
    return [-np.sin(I2)*(dtmd/2),-np.sin(I2)*(dtmd/2)]

sss = Solver()
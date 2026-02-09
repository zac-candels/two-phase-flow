import numpy as np
import NSAC

def runDropletSim():
    
    R0 = 2
    L_x = 5*R0
    L_y = 2*R0
    nx = 80
    ny = 60
    beta = 0.00000001
    Pe = 1
    Re = 0.1
    Cn = 0.05
    We = 1
    theta_E = 60
    initShape = "circle"
    testType = "equil"
    xc = L_x/2
    if testType == "measureCA":
        yc = R0 - 0.6*R0
    elif testType == "square":
        yc = L_y/2
    elif testType == "equil":
        yc = - R0*np.sqrt( np.tan(theta_E*np.pi/180)**2 + 1)\
            / (np.tan(theta_E*np.pi/180)**2 + 1)
    dataDir = "test_" + testType
    
    print("yc = ", yc)
    NSAC.dropletSim(theta_E, L_x, L_y, xc, yc, nx, ny, R0, Cn,
                    We, Re, Pe, beta, initShape, testType, dataDir)

runDropletSim()
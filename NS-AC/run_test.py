import numpy as np
import NSAC

def runDropletSim():

    # Spreading dynamics, vary Peclet number logarithmically. See where things are stable
    # Push droplet, see if things look ok. This means applying a horizontal body force to NS equations.
    # Start with \theta_D = \theta_E, see if things change (they shouldn't).
    
    R0 = 2
    L_x = 8*R0
    L_y = 2*R0
    nx = 80
    ny = 60
    beta = 0.00000001
    Pe = 0.1275
    Re = 0.1
    Cn = 0.05
    We = 1
    bodyForceMag = 0
    theta_E = 150
    initShape = "circle"
    testType = "measureCA"
    xc = L_x/2

    if testType == "measureCA":
        yc = R0 - 0.6*R0
    elif testType == "square":
        yc = L_y/2
    elif testType == "equil":
        yc = - R0*np.sqrt( np.tan(theta_E*np.pi/180)**2 + 1)\
            / (np.tan(theta_E*np.pi/180)**2 + 1)
    else:
        yc = R0 - 0.6*R0

    if testType == "bodyForce":
        bodyForceMag = 10

    dataDir = testType + "_Pe" + str(Pe) + "_Cn" + str(Cn) + "_We" + str(We) + "_Re" + str(Re) + "_CA" + str(theta_E)
    
    print("yc = ", yc)
    NSAC.dropletSim(theta_E, L_x, L_y, xc, yc, nx, ny, R0, Cn,
                    We, Re, Pe, beta, bodyForceMag, initShape, testType, dataDir)

runDropletSim()
# This is just an own "Learn" Davidson algorithm, no optimizations of the code are taken into account 
# ref: M. Crouzeix, B. Philippe, and M. Sadkane, SIAM J. Sci. Comput. (USA) 15(1), 62 (1994).
#
# numpy references: https://numpy.org/doc/stable/reference/generated/numpy.linalg.eig.html
# N. Foglia 06/2022

import math
import numpy as np

# Parameters
Dim = 1000              # Dimension of matrix to obtain eingpairs
tol = 1e-5              # Convergence tolerance
density = 0.000000001      # --> 0 for sparse matrix; --> infty for dense matrix
kin = 15                # number of initial guess vectors 
keig = 10               # number of eignpairs to obtain 
Miter = 5               # max number of iterations
minmax = 1              # 1 for estimate minimum eigen states, -1 for maximun eigen states

############################################
# Generate a random sparse H matrix to obtain its eigen pairs
############################################
H = np.zeros((Dim,Dim))
#Main diagonal components
for i in range(0,Dim):
    H[i,i] = 2*i
#add offdiagonal elements
H = H + density*np.random.randn(Dim,Dim)
#ensure Hermiticity
H = (H.T + H)/2

############################################
# Now do Davidson
############################################
V = np.eye(Dim,Dim)             # array to hold guess vec
Id = np.eye(Dim)                # identity matrix same dimen as A

for iter in range(0,Miter):
    print("Iter:",iter)

    # Compute W_iter = H * V_iter
    Wi = np.dot(H,V[:,:kin])
    # Compute Rayleigh Matrix (H projected in V subspace) H_iter = V_iter^T * W_iter = V_iter^T * H * V_iter
    Hi = np.dot(V[:,:kin].T,Wi)
    # Diagonalize H_iter and obtain eigenpairs (lambda,y)
    Lambda, y = np.linalg.eig(Hi)
    # Order eigenvalues in ascendig/descending order acording to minmax sign
    index = (minmax*Lambda).argsort()
    Lambda = Lambda[index]
    y = y[:,index]
    # Check convergence for the first keig eigen pairs
    converged = True
    if (iter == 0):
        converged = False

    addedvectors = 0
    for ivec in range(0,keig):
        # Compute Ritz vector x = V_iter * y
        x = np.dot(V[:,:kin],y[:,ivec])
        # Compute the residual r = Lambda * x - Wi * y
        r = Lambda[ivec] * x - np.dot(Wi,y[:,ivec])
        # Check convergence
        converged = converged & (np.linalg.norm(r) < tol)
        # Compute new direction
        # Clasical Davidson C = (lambda * Id - Hdiagonal) ^-1
        Cp = (Lambda[ivec] - H[ivec,ivec])

        if (abs(Cp) > 0.000000000001):
            r = r / abs(Cp)
            # Ortogonalize r
            for jvec in range(0,keig):
                r = r - np.dot(r,V[:,jvec])*V[:,jvec]
                # Normalize r    
                r = r / np.linalg.norm(r)
                # add vector
                V[:,(kin+addedvectors)] = r
                addedvectors += 1

    kin += addedvectors
    if converged:
            break


print("#####################################################")
print("My Davidson")
print("#####################################################")
for ivec in range(0,keig):
    print("eigenpair:", Lambda[ivec])
#    print(y[:,ivec])


############################################
# Compare with a full diagonalization
############################################
print("#####################################################")
print("Full diagonalization")
print("#####################################################")

Lambda, y = np.linalg.eig(H)
# Order eigenvalues in ascendig/descending order acording to minmax sign
index = (minmax*Lambda).argsort()
Lambda = Lambda[index]
y = y[:,index]

for ivec in range(0,keig):
    print("eigenpair:", Lambda[ivec])
#    print(y[:,ivec])

                                     

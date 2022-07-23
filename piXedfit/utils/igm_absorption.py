import numpy as np

__all__ = ["igm_att_madau", "tau_LAF_LS", "tau_DLA_LS", "tau_DLA_LC", "tau_LAF_LC", "igm_att_inoue"]

#==> IGM absorption based on Madau+1995:

# wave: wavelength in Angstroms
def igm_att_madau(wave,z):
	wave = np.asarray(wave)
	nwaves = len(wave)

	lylambda = np.zeros(32)
	madauA = np.zeros(32)

	nseries = 4
	for ii in range(2,int(nseries)+2):
		lylambda[ii-2] = 912.0*np.power(ii,2.0)/(np.power(ii,2.0)-1.0)

	madauA[0] = 0.0036
	madauA[1] = 0.0017
	madauA[2] = 0.0012
	madauA[3] = 0.00093

	teffline = np.zeros(nwaves)
	for ii in range(0,nseries):
		lmin = lylambda[ii]
		lmax = lylambda[ii]*(1.0+z)

		idx0 = np.where((wave>=lmin) & (wave<=lmax))
		teffline[idx0[0]] += madauA[ii]*np.exp(3.46*np.log(wave[idx0[0]]/lylambda[ii]))

	xc = wave/912.0
	xem = 1.0+z

	idx = np.where(xc<1.0)
	xc[idx[0]] = 1.0

	idx = np.where(xc>xem)
	xc[idx[0]] = xem

	teffcont = 0.25*xc*xc*xc*(np.exp(0.46*np.log(xem)) - np.exp(0.46*np.log(xc)))
	teffcont = teffcont + 9.4*np.exp(1.5*np.log(xc))*(np.exp(0.18*np.log(xem)) - np.exp(0.18*np.log(xc)))
	teffcont = teffcont - 0.7*xc*xc*xc*(np.exp(-1.32*np.log(xc)) - np.exp(-1.32*np.log(xem)))
	teffcont = teffcont - 0.023*(np.exp(1.68*np.log(xem)) - np.exp(1.68*np.log(xc)))
	tefftot = teffline + teffcont

	return np.exp(-1.0*tefftot)


#==> IGM absorption based on Inoue+2014:

def tau_LAF_LS(wave,z):
	# based on table 2 of inoue et al. 2014
	nj = 39
	lj  = np.asarray([1215.67, 1025.72, 972.537, 949.743, 937.803, 930.748, 926.226, 923.150, 920.963, 919.352, 918.129, 917.181, 916.429, 915.824, 915.329, 914.919, 914.576, 914.286, 914.039, 913.826, 913.641, 913.480, 913.339, 913.215, 913.104, 913.006, 912.918, 912.839, 912.768, 912.703, 912.645, 912.592, 912.543, 912.499, 912.458, 912.420, 912.385, 912.353, 912.324])
	Aj1 = np.asarray([1.690e-2,4.692e-3,2.239e-3,1.319e-3,8.707e-4,6.178e-4,4.609e-4,3.569e-4,2.843e-4,2.318e-4,1.923e-4,1.622e-4,1.385e-4,1.196e-4,1.043e-4,9.174e-5,8.128e-5,7.251e-5,6.505e-5,5.868e-5,5.319e-5,4.843e-5,4.427e-5,4.063e-5,3.738e-5,3.454e-5,3.199e-5,2.971e-5,2.766e-5,2.582e-5,2.415e-5,2.263e-5,2.126e-5,2.000e-5,1.885e-5,1.779e-5,1.682e-5,1.593e-5,1.510e-5])
	Aj2 = np.asarray([2.354e-3,6.536e-4,3.119e-4,1.837e-4,1.213e-4,8.606e-5,6.421e-5,4.971e-5,3.960e-5,3.229e-5,2.679e-5,2.259e-5,1.929e-5,1.666e-5,1.453e-5,1.278e-5,1.132e-5,1.010e-5,9.062e-6,8.174e-6,7.409e-6,6.746e-6,6.167e-6,5.660e-6,5.207e-6,4.811e-6,4.456e-6,4.139e-6,3.853e-6,3.596e-6,3.364e-6,3.153e-6,2.961e-6,2.785e-6,2.625e-6,2.479e-6,2.343e-6,2.219e-6,2.103e-6])
	Aj3 = np.asarray([1.026e-4,2.849e-5,1.360e-5,8.010e-6,5.287e-6,3.752e-6,2.799e-6,2.167e-6,1.726e-6,1.407e-6,1.168e-6,9.847e-7,8.410e-7,7.263e-7,6.334e-7,5.571e-7,4.936e-7,4.403e-7,3.950e-7,3.563e-7,3.230e-7,2.941e-7,2.689e-7,2.467e-7,2.270e-7,2.097e-7,1.943e-7,1.804e-7,1.680e-7,1.568e-7,1.466e-7,1.375e-7,1.291e-7,1.214e-7,1.145e-7,1.080e-7,1.022e-7,9.673e-8,9.169e-8])

	tau = np.zeros(len(wave))
	for jj in range(0,int(nj)):
		# eqn 21 of inoue et al. 2014
		idx0 = np.where((wave>lj[jj]) & (wave<lj[jj]*(1.0+z)) & (wave<2.2*lj[jj]))
		tau[idx0[0]] += Aj1[jj]*np.power(wave[idx0[0]]/lj[jj],1.2)

		idx1 = np.where((wave>lj[jj]) & (wave<lj[jj]*(1.0+z)) & (wave>=2.2*lj[jj]) & (wave<5.7*lj[jj]))
		tau[idx1[0]] += Aj2[jj]*np.power(wave[idx1[0]]/lj[jj],3.7)

		idx2 = np.where((wave>lj[jj]) & (wave<lj[jj]*(1.0+z)) & (wave>=5.7*lj[jj]))
		tau[idx2[0]] += Aj3[jj]*np.power(wave[idx2[0]]/lj[jj],5.5)

	return tau


def tau_DLA_LS(wave,z):
	# table 2 of inoue et al. 2014
	nj = 39
	lj = np.asarray([1215.67, 1025.72, 972.537, 949.743, 937.803, 930.748, 926.226, 923.150, 920.963, 919.352, 918.129, 917.181, 916.429, 915.824, 915.329, 914.919, 914.576, 914.286, 914.039, 913.826, 913.641, 913.480, 913.339, 913.215, 913.104, 913.006, 912.918, 912.839, 912.768, 912.703, 912.645, 912.592, 912.543, 912.499, 912.458, 912.420, 912.385, 912.353, 912.324])
	Aj1 = np.asarray([1.617e-4,1.545e-4,1.498e-4,1.460e-4,1.429e-4,1.402e-4,1.377e-4,1.355e-4,1.335e-4,1.316e-4,1.298e-4,1.281e-4,1.265e-4,1.250e-4,1.236e-4,1.222e-4,1.209e-4,1.197e-4,1.185e-4,1.173e-4,1.162e-4,1.151e-4,1.140e-4,1.130e-4,1.120e-4,1.110e-4,1.101e-4,1.091e-4,1.082e-4,1.073e-4,1.065e-4,1.056e-4,1.048e-4,1.040e-4,1.032e-4,1.024e-4,1.017e-4,1.009e-4,1.002e-4])
	Aj2 = np.asarray([5.390e-5,5.151e-5,4.992e-5,4.868e-5,4.763e-5,4.672e-5,4.590e-5,4.516e-5,4.448e-5,4.385e-5,4.326e-5,4.271e-5,4.218e-5,4.168e-5,4.120e-5,4.075e-5,4.031e-5,3.989e-5,3.949e-5,3.910e-5,3.872e-5,3.836e-5,3.800e-5,3.766e-5,3.732e-5,3.700e-5,3.668e-5,3.637e-5,3.607e-5,3.578e-5,3.549e-5,3.521e-5,3.493e-5,3.466e-5,3.440e-5,3.414e-5,3.389e-5,3.364e-5,3.339e-5])

	tau = np.zeros(len(wave))
	for jj in range(0,int(nj)):
		# eqn 22 of inoue et al. 2014
		idx0 = np.where((wave>lj[jj]) & (wave<lj[jj]*(1.0+z)) & (wave<3.0*lj[jj]))
		tau[idx0[0]] += Aj1[jj]*np.power(wave[idx0[0]]/lj[jj],2.0)

		idx0 = np.where((wave>lj[jj]) & (wave<lj[jj]*(1.0+z)) & (wave>=3.0*lj[jj]))
		tau[idx0[0]] += Aj2[jj]*np.power(wave[idx0[0]]/lj[jj],3.0)

	return tau


def tau_DLA_LC(wave,z):
	lL = 911.8   # Lyman limit

	# eqn 28 and 29 of Inoue et al. 2014
	tau = np.zeros(len(wave))
	if z<2.0:
		idx0 = np.where(wave<lL*(1+z))
		tau[idx0[0]] = 0.211*np.power(1.0+z,2) - 7.66e-2*np.power(1.0+z,2.3)*np.power(wave[idx0[0]]/lL,-0.3) - 0.135*np.power(wave[idx0[0]]/lL,2.0)

		idx1 = np.where(wave>=lL*(1+z))
		tau[idx1[0]] = 0.0

	else:
		idx0 = np.where(wave<3.0*lL)
		tau[idx0[0]] = 0.634 + 4.70e-2*np.power(1.0+z,3.0) - 1.78e-2*np.power(1.0+z,3.3)*np.power(wave[idx0[0]]/lL,-0.3) - 0.135*np.power(wave[idx0[0]]/lL,2.0) - 0.291*np.power(wave[idx0[0]]/lL,-0.3)

		idx1 = np.where((wave>=3.0*lL) & (wave<lL*(1.0+z)))
		tau[idx1[0]] = 4.70e-2*np.power(1.0+z,3.0) - 1.78e-2*np.power(1.0+z,3.3)*np.power(wave[idx1[0]]/lL,-0.3) - 2.92e-2*np.power(wave[idx1[0]]/lL,3)

		idx2 = np.where(wave>=lL*(1.0+z))
		tau[idx2[0]] = 0.0

	return tau


def tau_LAF_LC(wave,z):
	# eqn 25, 26 and 29 of Inoue et al. 2014:
	lL = 911.8   # Lyman limit

	tau = np.zeros(len(wave))

	if z<1.2:
		idx0 = np.where(wave<lL*(1+z))
		tau[idx0[0]] = 0.325*(np.power(wave[idx0[0]]/lL,1.2)-np.power(1.0+z,-0.9)*np.power(wave[idx0[0]]/lL,2.1))

		idx1 = np.where(wave>=lL*(1+z))
		tau[idx1[0]] = 0.0

	elif z<4.7:
		idx0 = np.where(wave<2.2*lL)
		tau[idx0[0]] = 2.55e-2*np.power(1.0+z,1.6)*np.power(wave[idx0[0]]/lL,2.1) + 0.325*np.power(wave[idx0[0]]/lL,1.2) - 0.250*np.power(wave[idx0[0]]/lL,2.1)

		idx1 = np.where((wave>=2.2*lL) & (wave<lL*(1+z)))
		tau[idx1[0]] = 2.55e-2*(np.power(1.0+z,1.6)*np.power(wave[idx1[0]]/lL,2.1) - np.power(wave[idx1[0]]/lL,3.7))

		idx2 = np.where(wave>=lL*(1+z))
		tau[idx2[0]] = 0.0
			
	else:
		idx0 = np.where(wave<2.2*lL)
		tau[idx0[0]] = 5.22e-4*np.power(1.0+z,3.4)*np.power(wave[idx0[0]]/lL,2.1) + 0.325*np.power(wave[idx0[0]]/lL,1.2) - 3.14e-2*np.power(wave[idx0[0]]/lL,2.1)

		idx1 = np.where((wave>=2.2*lL) & (wave<5.7*lL))
		tau[idx1[0]] = 5.22e-4*np.power(1.0+z,3.4)*np.power(wave[idx1[0]]/lL,2.1) + 0.218*np.power(wave[idx1[0]]/lL,2.1) - 2.55e-2*np.power(wave[idx1[0]]/lL,3.7)

		idx2 = np.where((wave>=5.7*lL) & (wave<lL*(1+z)))
		tau[idx2[0]] = 5.22e-4*(np.power(1.0+z,3.4)*np.power(wave[idx2[0]]/lL,2.1) - np.power(wave[idx2[0]]/lL,5.5))

		idx3 = np.where(wave>=lL*(1+z))
		tau[idx3[0]] = 0.0

	return tau


def igm_att_inoue(wave,z):
	# eq. 15 of Inoue + 2014:
	tau = tau_LAF_LS(wave,z) + tau_DLA_LS(wave,z) + tau_LAF_LC(wave,z) + tau_DLA_LC(wave,z)

	return np.exp(-1.0*tau)


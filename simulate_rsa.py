import numpy as np
import matplotlib.pyplot as plt

plt.interactive(True)


def compute_ABC(X):
    A = np.abs(np.subtract.outer(X[:, 0], X[:, 0]))
    B = np.abs(np.subtract.outer(X[:, 1], X[:, 1]))
    C = np.abs(np.subtract.outer(X[:, 2], X[:, 2]))
    return A, B, C


def compute_rhos(A, B, C):
    triu_idx = np.triu_indices(A.shape[0], k=1)
    rho_AB = np.corrcoef(A[triu_idx], B[triu_idx])[0, 1]
    rho_AC = np.corrcoef(A[triu_idx], C[triu_idx])[0, 1]
    rho_BC = np.corrcoef(B[triu_idx], C[triu_idx])[0, 1]
    return rho_AB, rho_AC, rho_BC


if __name__ == '__main__':

    np.random.seed(0)

    plot = True

    n_samples = 100
    n_permutations = 1000

    print("These are the correlations among the three variables a, b and c:")
    r_ab = 0.0
    r_ac = 0.5
    r_bc = 0.5
    print("r_ab = %f" % r_ab)
    print("r_ac = %f" % r_ac)
    print("r_bc = %f" % r_bc)

    print("This is the covariace matrix:")
    cov = np.array([[1, r_ab, r_ac],
                    [r_ab, 1, r_bc],
                    [r_ac, r_bc, 1]], dtype=np.float)
    print(cov)

    # This ensures that cov is positive definite (See for example Rose
    # and Smith 2002, Mathematical Statistics with Mathematica,
    # ch.6.4, p.226):
    assert((r_ab >= -1.0) and (r_ab <= 1.0))
    assert((r_ac >= -1.0) and (r_ac <= 1.0))
    assert((r_bc >= -1.0) and (r_bc <= 1.0))
    det_cov = np.linalg.det(cov)
    print("det(cov) = %f" % det_cov)
    assert(det_cov > 0.0)

    print("Sampling %d points from the multivariate_normal(0,cov)" % n_samples)
    X = np.random.multivariate_normal(mean=np.zeros(3), cov=cov,
                                      size=n_samples)

    A, B, C = compute_ABC(X)

    rho_AB, rho_AC, rho_BC = compute_rhos(A, B, C)
    print("rho_AB = %f" % rho_AB)
    print("rho_AC = %f" % rho_AC)
    print("rho_BC = %f" % rho_BC)

    print("Starting %d permutations." % n_permutations)
    rho_AB_permuted = np.zeros(n_permutations)
    rho_AC_permuted = np.zeros(n_permutations)
    rho_BC_permuted = np.zeros(n_permutations)
    for i in range(n_permutations):
        idxA = np.random.permutation(n_samples)
        A_permuted = A[idxA, :][:, idxA]
        idxB = np.random.permutation(n_samples)
        B_permuted = A[idxB, :][:, idxB]
        idxC = np.random.permutation(n_samples)
        C_permuted = A[idxC, :][:, idxC]
        # B_permuted, C_permuted = B, C
        rho_AB_permuted[i], \
            rho_AC_permuted[i], \
            rho_BC_permuted[i] = compute_rhos(A_permuted,
                                              B_permuted,
                                              C_permuted)
        # print("%d) \t %0.5f \t %0.5f \t %0.5f" % (i, rho_AB_permuted[i],
        #                                           rho_AC_permuted[i],
        #                                           rho_BC_permuted[i]))

    p_value_AB = (rho_AB <= rho_AB_permuted).sum() \
                 / float(len(rho_AB_permuted))
    p_value_AC = (rho_AC <= rho_AC_permuted).sum() \
                 / float(len(rho_AC_permuted))
    p_value_BC = (rho_BC <= rho_BC_permuted).sum() \
                 / float(len(rho_BC_permuted))
    print("p_value_AB = %f" % p_value_AB)
    print("p_value_AC = %f" % p_value_AC)
    print("p_value_BC = %f" % p_value_BC)

    plt.figure()
    plt.subplot(131)
    plt.hist(rho_AB_permuted, bins=50)
    plt.subplot(132)
    plt.hist(rho_AC_permuted, bins=50)
    plt.subplot(133)
    plt.hist(rho_BC_permuted, bins=50)

    if plot:
        plt.figure()
        plt.subplot(131)
        plt.plot(X[:, 0], X[:, 1], 'ko')
        plt.axis('equal')
        plt.xlabel('A')
        plt.ylabel('B')
        plt.subplot(132)
        plt.plot(X[:, 0], X[:, 2], 'ko')
        plt.axis('equal')
        plt.xlabel('A')
        plt.ylabel('C')
        plt.subplot(133)
        plt.plot(X[:, 1], X[:, 2], 'ko')
        plt.axis('equal')
        plt.xlabel('B')
        plt.ylabel('C')

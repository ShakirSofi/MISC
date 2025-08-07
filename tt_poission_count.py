import numpy as np
from scipy.optimize import minimize
from scipy.special import factorial

# --- Helper: TT reconstruction (small scale) ---
def tt_reconstruct(Gs):
    # Gs = list of TT cores, each shape (r_k-1, n_k, r_k)
    # Multiply cores to reconstruct full tensor
    # For 3D tensor: G1 (1, n1, r1), G2 (r1, n2, r2), G3 (r2, n3, 1)
    G1, G2, G3 = Gs
    n1 = G1.shape[1]
    n2 = G2.shape[1]
    n3 = G3.shape[1]
    r1 = G1.shape[2]
    r2 = G2.shape[2]

    tensor = np.zeros((n1, n2, n3))
    for i in range(n1):
        for j in range(n2):
            for k in range(n3):
                # Contract TT cores for element (i,j,k)
                val = 0
                for alpha in range(r1):
                    for beta in range(r2):
                        val += G1[0,i,alpha] * G2[alpha,j,beta] * G3[beta,k,0]
                tensor[i,j,k] = val
    return tensor

# --- Negative log-likelihood ---
def neg_log_likelihood(params, shape, rank, X):
    # Unpack TT cores from flat params vector
    n1, n2, n3 = shape
    r1, r2 = rank

    size_G1 = 1 * n1 * r1
    size_G2 = r1 * n2 * r2
    size_G3 = r2 * n3 * 1

    G1 = params[0:size_G1].reshape((1, n1, r1))
    G2 = params[size_G1:size_G1+size_G2].reshape((r1, n2, r2))
    G3 = params[size_G1+size_G2:].reshape((r2, n3, 1))

    Gs = [G1, G2, G3]

    # Reconstruct tensor and apply exp for positivity
    log_lambda = tt_reconstruct(Gs)
    lambda_ = np.exp(log_lambda)

    # Avoid zero or negative lambda for numerical stability
    lambda_ = np.clip(lambda_, 1e-10, None)

    # Negative log likelihood (ignoring log factorial term)
    nll = np.sum(lambda_ - X * np.log(lambda_))
    return nll

# --- Generate synthetic count tensor ---
np.random.seed(42)
n1, n2, n3 = 4, 4, 4
true_lambda = 3.0  # constant rate for simplicity
X = np.random.poisson(lam=true_lambda, size=(n1,n2,n3))

# --- Initialization of TT cores ---
rank = (2, 2)
G1_init = np.random.randn(1, n1, rank[0]) * 0.1
G2_init = np.random.randn(rank[0], n2, rank[1]) * 0.1
G3_init = np.random.randn(rank[1], n3, 1) * 0.1

params_init = np.concatenate([G1_init.ravel(), G2_init.ravel(), G3_init.ravel()])

# --- Optimize ---
result = minimize(
    neg_log_likelihood,
    params_init,
    args=((n1, n2, n3), rank, X),
    method='L-BFGS-B',
    options={'maxiter': 1000, 'disp': True}
)

# --- Extract optimized TT cores ---
opt_params = result.x
G1_opt = opt_params[0:1*n1*rank[0]].reshape((1, n1, rank[0]))
G2_opt = opt_params[1*n1*rank[0]:1*n1*rank[0]+rank[0]*n2*rank[1]].reshape((rank[0], n2, rank[1]))
G3_opt = opt_params[1*n1*rank[0]+rank[0]*n2*rank[1]:].reshape((rank[1], n3, 1))

# Reconstruct final lambda
lambda_est = np.exp(tt_reconstruct([G1_opt, G2_opt, G3_opt]))

print("Original count tensor (X):")
print(X)
print("\nEstimated lambda tensor:")
print(lambda_est)

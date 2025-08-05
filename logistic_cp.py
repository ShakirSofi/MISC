import numpy as np

def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))

def logistic_cp_loss_grad(Y, U, V, W, bias, lam):
    """
    Compute logistic loss and gradients for CP factors.
    Y: binary tensor (n x m x p) with 0/1 entries.
    U: (n x R), V: (m x R), W: (p x R) factor matrices.
    bias: scalar global bias term.
    lam: L2 regularization weight.
    Returns: loss, grad_U, grad_V, grad_W, grad_bias.
    """
    n, m, p = Y.shape
    R = U.shape[1]
    # Compute full logit tensor Theta = bias + sum_r outer(U[:,r], V[:,r], W[:,r])
    Theta = np.zeros((n, m, p))
    for r in range(R):
        # Compute outer product of vectors U[:,r], V[:,r], W[:,r]
        # We can do this by np.tensordot or nested outer.
        Theta += np.einsum('i,j,k->ijk', U[:,r], V[:,r], W[:,r])
    Theta += bias  # add bias to all entries

    # Compute logistic loss: sum(log(1+exp(Theta)) - Y*Theta) + reg
    logexp = np.log1p(np.exp(Theta))
    loss = np.sum(logexp - Y * Theta)
    # Add L2 regularization
    loss += lam*(np.sum(U*U) + np.sum(V*V) + np.sum(W*W))

    # Compute residuals E = sigmoid(Theta) - Y
    P = sigmoid(Theta)
    E = P - Y  # shape (n,m,p)

    # Gradients initialization
    grad_U = np.zeros_like(U)
    grad_V = np.zeros_like(V)
    grad_W = np.zeros_like(W)

    # Compute gradients for each rank component
    for r in range(R):
        # Outer products for gradient computation
        # gradient_U[:,r] = sum_{j,k} E * (V[j,r]*W[k,r]) for each i
        outer_VW = np.outer(V[:,r], W[:,r])  # shape (m,p)
        grad_U[:,r] = np.tensordot(E, outer_VW, axes=([1,2],[0,1]))
        # grad_V[:,r] = sum_{i,k} E * (U[i,r]*W[k,r]) for each j
        outer_UW = np.outer(U[:,r], W[:,r])  # (n,p)
        grad_V[:,r] = np.tensordot(E, outer_UW, axes=([0,2],[0,1]))
        # grad_W[:,r] = sum_{i,j} E * (U[i,r]*V[j,r]) for each k
        outer_UV = np.outer(U[:,r], V[:,r])  # (n,m)
        grad_W[:,r] = np.tensordot(E, outer_UV, axes=([0,1],[0,1]))

    # Add gradient of L2 regularization
    grad_U += 2 * lam * U
    grad_V += 2 * lam * V
    grad_W += 2 * lam * W

    # Gradient for bias term = sum of residuals
    grad_bias = np.sum(E)

    return loss, grad_U, grad_V, grad_W, grad_bias


def logistic_cp_loss_grad_confidence(Y, C, U, V, W, bias, lam):
    """
    Logistic CP loss and gradients with confidence weights.
    
    Parameters:
        Y: binary tensor (n x m x p)
        C: confidence weights tensor (n x m x p)
        U, V, W: factor matrices (n x R, m x R, p x R)
        bias: scalar
        lam: L2 regularization weight

    Returns:
        loss, grad_U, grad_V, grad_W, grad_bias
    """
    n, m, p = Y.shape
    R = U.shape[1]

    # Compute Theta = sum_r U[:,r] ⊗ V[:,r] ⊗ W[:,r] + bias
    Theta = np.einsum('ir,jr,kr->ijk', U, V, W) + bias

    # P = sigmoid(Theta)
    P = sigmoid(Theta)

    # Compute confidence-weighted logistic loss
    logexp = np.log1p(np.exp(Theta))  # log(1 + exp(Theta))
    loss = np.sum(C * (logexp - Y * Theta))
    loss += lam * (np.sum(U**2) + np.sum(V**2) + np.sum(W**2))

    # Compute residuals: C * (P - Y)
    E = C * (P - Y)  # shape (n,m,p)

    # Gradients using einsum (more efficient and readable)
    grad_U = np.einsum('ijk,jr,kr->ir', E, V, W) + 2 * lam * U
    grad_V = np.einsum('ijk,ir,kr->jr', E, U, W) + 2 * lam * V
    grad_W = np.einsum('ijk,ir,jr->kr', E, U, V) + 2 * lam * W
    grad_bias = np.sum(E)

    return loss, grad_U, grad_V, grad_W, grad_bias
    

n, m, p = (3, 4, 5)   # dimensions (users x items x contexts)
R = 2               # chosen rank
# Create random ground-truth factors and bias, then generate Y
U_true = np.random.randn(n, R)
V_true = np.random.randn(m, R)
W_true = np.random.randn(p, R)
bias_true = 0.5
# Construct logits Theta_true
Theta_true = np.einsum('ir,jr,kr->ijk', U_true, V_true, W_true)
Theta_true += bias_true
# Generate binary data via logistic model
Y_prob = sigmoid(Theta_true)
Y = (np.random.rand(n, m, p) < Y_prob).astype(float)

# Initialize factors for learning
U = np.random.randn(n, R) * 0.1
V = np.random.randn(m, R) * 0.1
W = np.random.randn(p, R) * 0.1
bias = 0.01
learning_rate = 0.05
reg = 0.05

# Gradient descent loop
for it in range(1000):
    loss, gU, gV, gW, gb = logistic_cp_loss_grad(Y, U, V, W, bias, reg)
    if it % 200 == 0:
        print(f"Iter {it:4d}, loss={loss:.3f}, bias={bias:.3f}")
    # Update parameters
    U -= learning_rate * gU
    V -= learning_rate * gV
    W -= learning_rate * gW
    bias -= learning_rate * gb

print(f"Final loss: {loss:.3f}, bias={bias:.3f}")

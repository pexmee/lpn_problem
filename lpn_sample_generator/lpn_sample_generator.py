import numpy as np

"""
    NOTE:
    Binomial distribution is a probability distribution used in statistics 
    that summarizes the likelihood that a value will take one of two independent
    values under a given set of parameters or assumptions.
"""

def generate_sample(s,p,amount):
    dim = len(s)

    # Randomly generate matrix A where each num ∈ {0,1}
    A = np.random.randint(0,2,size=(amount,dim))

    # Add errors using binominal distribution
    e = np.random.binomial(1, p, amount)

    # Multiply matrix A with secret key s and add error e
    As_with_err = (A @ s) + e

    # b is the element-wise remainder of matrix division (As+e)/2, i.e (As+e)%2
    b = np.mod(As_with_err, 2)
    return A,b

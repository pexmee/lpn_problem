import numpy as np
from threading import Thread
from typing import List

"""
    NOTE:
    Binomial distribution is a probability distribution used in statistics 
    that summarizes the likelihood that a value will take one of two independent
    values under a given set of parameters or assumptions.
"""


def generate_sample(s, p, amount):
    dim = len(s)

    # Randomly generate matrix A where each num ∈ {0,1}
    A = np.random.randint(0, 2, size=(amount, dim))

    # Add errors using binominal distribution
    e = np.random.binomial(1, p, amount)

    # Multiply matrix A with secret key s and add error e
    As_with_err = (A @ s) + e

    # b is the element-wise remainder of matrix division (As+e)/2, i.e (As+e)%2
    b = np.mod(As_with_err, 2)
    return A, b


def generate_wrapper(s, p, amount, results, i):
    results[i] = generate_sample(s, p, amount)


def generate_multiple_samples(s, p, amount, num_samples):
    threads: List[Thread] = []
    results = [None] * num_samples

    # Generate samples by calling wrapper
    for i in range(num_samples):
        th = Thread(target=generate_wrapper, args=(s, p, amount, results, i))
        th.start()
        threads.append(th)

    # Sync threads
    for thread in threads:
        thread.join()

    return results


results = generate_multiple_samples(np.random.randint(0, 2, 16), 0.125, 100, 10)
for i, result in enumerate(results):
    print(f"sample #{i}:\n A: {result[0]}\n b: {result[1]}")

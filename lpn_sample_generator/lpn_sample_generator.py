from threading import Thread
from typing import List, Tuple

import numpy as np
from numpy.typing import NDArray
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier


def generate_sample(s: NDArray, p: float, size: int) -> Tuple[NDArray, NDArray]:
    dim = len(s)

    # Randomly generate matrix A where each num ∈ {0,1}
    A = np.random.randint(0, 2, size=(size, dim))

    # Add errors using binominal distribution
    e = np.random.binomial(1, p, size)

    # Multiply matrix A with secret key s and add error e
    As_with_err = (A @ s) + e

    # b is the element-wise remainder of matrix division (As+e)/2, i.e (As+e)%2
    b = np.mod(As_with_err, 2)
    return A, b


def generate_wrapper(s, p, size, results, i):
    results[i] = generate_sample(s, p, size)


def generate_multiple_samples(s, p, size, num_samples):
    threads: List[Thread] = []
    results = [None] * num_samples

    # Generate samples by calling wrapper
    for i in range(num_samples):
        th = Thread(target=generate_wrapper, args=(s, p, size, results, i))
        th.start()
        threads.append(th)

    # Sync threads
    for thread in threads:
        thread.join()

    return results


def decision_tree_classifier(A, b, dim, s):
    decision_tree = DecisionTreeClassifier()
    # Fit the tree.
    decision_tree.fit(A, b)
    # Make prediction for secret
    candidate_secret = decision_tree.predict(np.eye(dim))
    # Check if the candidate solution is correct.

    remainder_sum = np.mod(A @ candidate_secret + b, 2).sum()
    print("decision tree remainder: ", remainder_sum)

    # Hamming Weight
    if remainder_sum < 140000:
        print("Candidate secret for decision tree classifier:")
        print(f"candidate: {candidate_secret}\nSecret: {s}")
    else:
        print("Wrong candidate for decision tree classifier. Recursing..")
        decision_tree_classifier(A, b, dim, s)


# Bootstrapping
# Aggregation step -> different between random forest, extra trees
def random_forest_classifier(A, b, dim, s):
    random_forest = RandomForestClassifier()
    random_forest.fit(A, b)
    # Make prediction for secret
    candidate_secret = random_forest.predict(np.eye(dim))

    # Check if the candidate solution is correct.
    remainder_sum = np.mod(A @ candidate_secret + b, 2).sum()
    print("random forest remainder: ", remainder_sum)

    # Hamming Weight
    if remainder_sum < 140000:
        print("Candidate secret for random forest classifier:")
        print(f"candidate: {candidate_secret}\nSecret: {s}")
    else:
        print("Wrong candidate for random forest classifier. Recursing..")
        random_forest_classifier(A, b, dim, s)


def extra_trees_classifier(A, b, dim, s):
    extra_trees = ExtraTreesClassifier()
    extra_trees.fit(A, b)

    # Make prediction
    candidate_secret = extra_trees.predict(np.eye(dim))

    # Check if it is the correct candidate secret
    remainder_sum = np.mod(A @ candidate_secret + b, 2).sum()
    print("extra trees remainder: ", remainder_sum)

    # Hamming Weight
    if remainder_sum < 140000:
        print("Candidate secret for extra trees classifier:")
        print(f"candidate: {candidate_secret}\nSecret: {s}")
    else:
        print("Wrong candidate for extra trees classifier. Recursing..")
        extra_trees_classifier(A, b, dim, s)


# results = generate_multiple_samples(np.random.randint(0, 2, 16), 0.125, 100, 10)
# for i, result in enumerate(results):
#     print(f"sample #{i}:\n A: {result[0]}\n b: {result[1]}")

dim = 16  # n
s = np.random.randint(0, 2, dim)
p = 0.125  # epsylon ->
size = 100000  # m = number of samples

A, b = generate_sample(s, p, size)
decision_tree_classifier(A, b, dim, s)
random_forest_classifier(A, b, dim, s)
extra_trees_classifier(A, b, dim, s)

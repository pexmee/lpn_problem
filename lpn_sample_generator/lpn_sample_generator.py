from threading import Thread
from typing import List, Tuple

import numpy as np
from numpy.typing import NDArray
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from threading import Thread
import json

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

    hamming_weight = np.mod(A @ candidate_secret + b, 2).sum()
    #print("decision tree remainder: ", hamming_weight)

    # Hamming Weight
    if hamming_weight < 14000:
        #print("Candidate secret for decision tree classifier:")
        #print(f"candidate: {candidate_secret}\nSecret: {s}")
        return hamming_weight

    else:
        return -1


# Bootstrapping
# Aggregation step -> different between random forest, extra trees
def random_forest_classifier(A, b, dim, s):
    random_forest = RandomForestClassifier()
    
    random_forest.fit(A, b)
    # Make prediction for secret
    candidate_secret = random_forest.predict(np.eye(dim))

    # Check if it is an acceptable candidate.
    hamming_weight = np.mod(A @ candidate_secret + b, 2).sum()
    #print("random forest remainder: ", hamming_weight)

    # Hamming Weight
    if hamming_weight < 14000:
        #print("Candidate secret for random forest classifier:")
        #print(f"candidate: {candidate_secret}\nSecret: {s}")
        return hamming_weight
        
    else:
        return -1


def extra_trees_classifier(A, b, dim, s):
    extra_trees = ExtraTreesClassifier()
    extra_trees.fit(A, b)

    # Make prediction
    candidate_secret = extra_trees.predict(np.eye(dim))

    # Check if it is an acceptable candidate secret
    hamming_weight = np.mod(A @ candidate_secret + b, 2).sum()

    # Hamming Weight
    if hamming_weight < 14000:
        #print("Candidate secret for extra trees classifier:")
        #print(f"candidate: {candidate_secret}\nSecret: {s}")
        return hamming_weight
    else:
        return -1
        

def generic_classifier(classifier, A,b,dim,s,result):
    classifier.fit(A,b)

    # Make prediction
    candidate_secret = classifier.predict(np.eye(dim))

    # Check if it is an acceptable candidate secret
    hamming_weight = np.mod(A @ candidate_secret + b, 2).sum()
    print(f"hamming_weight: {hamming_weight}, int_hamming_weight: {int(hamming_weight)}")

    # Hamming Weight
    result.append(int(hamming_weight))
    # m = 4*dim(1/2 - p)^-2
    # t = m*p +sqrt(dim*m)
    # if hamming_weight <= t: # 
        # result.append(hamming_weight)
    
    # else:
        # result.append(hamming_weight)
    
        

# Generate a list of length n, containing random secrets of dimension dim
def generate_secrets(n,dim):
    return [np.random.randint(0,2,dim) for _ in range(n)]

def get_results(secrets, p, num_samples, dim):
    results = {"decision_class": [], "random_class" : [], "extra_class": []}
    
    threads = []
    
    for i in range(len(secrets)):
        print(f"Generating result number #{i+1}")
        decision_class = []
        random_class = []
        extra_class = []

        A,b = generate_sample(secrets[i],p,num_samples)
        for classifier,lst in zip([DecisionTreeClassifier,RandomForestClassifier,ExtraTreesClassifier],[decision_class,random_class,extra_class]):
            th = Thread(target=generic_classifier, args=(classifier(),A,b,dim,secrets[i],lst))
            th.start()
            threads.append(th)
        
        for th in threads:
            th.join()

        results["decision_class"]+=decision_class
        results["random_class"]+=random_class
        results["extra_class"]+=extra_class
        # dt = Thread(target=generic_classifier,args=(DecisionTreeClassifier(),A,b,dim,secrets[i],decision_class))
        # rt = Thread(target=generic_classifier,args=(RandomForestClassifier(),A,b,dim,secrets[i],random_class))
        # et = Thread(target=generic_classifier,args=(ExtraTreesClassifier(),A,b,dim,secrets[i],extra_class))

        # results["decision_class"].append(generic_classifier(DecisionTreeClassifier(),A,b,dim,secrets[i],decision_class))
        # results["random_class"].append(generic_classifier(RandomForestClassifier(),A,b,dim,secrets[i],random_class))
        # results["extra_class"].append(generic_classifier(ExtraTreesClassifier(),A,b,dim,secrets[i],extra_class))

    return results


        
        
# results = generate_multiple_samples(np.random.randint(0, 2, 16), 0.125, 100, 10)
# for i, result in enumerate(results):
#     #print(f"sample #{i}:\n A: {result[0]}\n b: {result[1]}")

num_secrets = 100
dim = 10 
s = np.random.randint(0, 2, dim)
p = 0.125  # epsylon ->
num_samples = 200000  # m = number of samples
A, b = generate_sample(s, p, num_samples)

results = get_results(generate_secrets(num_secrets,dim),p,num_samples,dim)
print(f"decision class: {results['decision_class']}")
print(f"random class: {results['random_class']}")
print(f"extra class: {results['extra_class']}")

with open("200_000_dim10.json","a") as file_h:
    json.dump(results,fp=file_h)

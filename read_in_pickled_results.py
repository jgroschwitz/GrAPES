import pickle
import sys

path_to_pickle = sys.argv[1]

results = pickle.load(open(path_to_pickle, "rb"))

print(results)
from joblib import Parallel, delayed
from tqdm import tqdm
import time

def process(i):
    time.sleep(0.01)
    return i

# Try return_as="generator"
try:
    results = list(tqdm(Parallel(n_jobs=2, return_as="generator")(delayed(process)(i) for i in range(10)), total=10, desc="Testing"))
    print("Success:", results)
except Exception as e:
    print("Failed:", e)

import time
import tqdm
import pandas
import matplotlib.pyplot as plt
import numpy as np
import itertools

def matmul(a,b):
    return a@b

obs= []
configs = list(itertools.product(
    [16, 32, 64, 128, 256],
    [np.float16, np.float32, np.float64]))

for dim, dtype in tqdm.tqdm(configs):
    for dtype in [np.float16, np.float32, np.float64]:
        a = np.random.rand(dim, dim).astype(dtype)
        b = np.random.rand(dim, dim).astype(dtype)
        matmul(a, b)

        begin = time.perf_counter()
        mes = []
        for i in range(100):
            begin = time.time()
            matmul(a ,b)
            duration = time.time() - begin
            mes.append(duration)
            if sum(mes) > 3:
                break
        obs.append(dict(
            dim=dim, 
            duration=sum(mes), 
            normalized = sum(mes) / len(mes) / dim**3, 
            max = max(mes), min = min(mes), 
            n_iter = len(mes), max_iter= mes.index(max(mes)),
            dtype= dtype.__name__))
        #print(f"dim={dim}, duration={duration}")

df = pandas.DataFrame(obs)
print(df)
fig, ax = plt.subplots(1,1)
df.plot(x="dim", y=["min", "normalized", "max"], ax=ax)
fig.savefig("bench_5.png")
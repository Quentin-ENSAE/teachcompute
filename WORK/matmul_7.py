import time
import tqdm
import pandas
import matplotlib.pyplot as plt
import numpy as np
import itertools

def matmul(a,b):
    return np.einsum("ij, jk -> ik", a, b)

obs= []
configs = list(itertools.product(
    [16, 32, 64, 128, 256],
    [np.float16, np.float32, np.float64]))

for dim, dtype in tqdm.tqdm(configs):
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
        duration=sum(mes)/len(mes), 
        normalized = sum(mes) / len(mes) / dim**3, 
        max = max(mes), min = min(mes), 
        n_iter = len(mes), max_iter= mes.index(max(mes)),
        dtype= dtype.__name__))
        #print(f"dim={dim}, duration={duration}")

df = pandas.DataFrame(obs)
piv = df.pivot(index="dim", columns="dtype", values="duration")
print(piv)
fig, ax = plt.subplots(1,2)
df.plot(x="dim", y=["min", "duration", "max"], ax=ax[0], 
        logx= True, logy=True)
piv.plot(ax=ax[1], logy=True, logx=True)
fig.savefig("bench_7.png")
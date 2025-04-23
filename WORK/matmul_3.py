import time
import tqdm
import pandas
import matplotlib.pyplot as plt
import numpy as np

def matmul(a,b):
    return a@b

obs= []
for dim in tqdm.tqdm([16, 32, 64, 128, 256]):
    a = np.random.rand(dim, dim)
    b = np.random.rand(dim, dim)
    matmul(a, b)

    begin = time.perf_counter()
    mes = []
    for i in range(100):
        begin = time.time()
        matmul(a ,b)
        duration = time.time() - begin
        mes.append(duration)
        if sum(mes) > 1:
            break
    obs.append(dict(dim=dim, duration=sum(mes), normalized = sum(mes) / len(mes) / dim**3, 
    max = max(mes), min = min(mes), 
    n_iter = len(mes), max_iter= mes.index(max(mes))))
    #print(f"dim={dim}, duration={duration}")

df = pandas.DataFrame(obs)
print(df)
fig, ax = plt.subplots(1,1)
df.plot(x="dim", y=["min", "normalized", "max"], ax=ax)
fig.savefig("bench_3.png")
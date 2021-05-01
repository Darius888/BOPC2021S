import matplotlib.pyplot as plt
import pandas as pd
import numpy as np


# EXPERIMENT 1 

mean_runtime = [0.9102,0.5109,0.3129,0.3129,0.1263,0.1263,0.1481,21.3172,10.8355,5.3968,2.7300,1.4382,1.0481,0.8584]

cores = [1,2,4,8,16,24,32,1,2,4,8,16,24,32]

rel_speed_up = [0,1.78,2.91,4.19,7.21,6.62,6.15,0,1.97,3.95,7.81,14.82,20.34,24.83]

par_eff = [0,0.89,0.73,0.52,0.45,0.28,0.19,0,0.99,0.99,0.98,0.93,0.85,0.78]

sizes = [200,200,200,200,200,200,200,1000,1000,1000,1000,1000,1000,1000]

run_time = pd.DataFrame(
    {'sizes': sizes,
     'mean_runtime': mean_runtime,
     'cores': cores
    })

df = run_time.pivot(index='cores', columns='sizes', values='mean_runtime')

df.plot()
plt.xticks(cores)
plt.ylabel("Run time (s)")
plt.show()



speed_up = pd.DataFrame(
    {
     'sizes': sizes,
     'rel_speed_up': rel_speed_up,
     'cores': cores
    })

df = speed_up.pivot(index='cores', columns='sizes', values='rel_speed_up')


df.plot()
plt.xticks(cores)
plt.ylabel("Speed up")
plt.show()



paralell_eff = pd.DataFrame(
    {
     'sizes': sizes,
     'par_eff': par_eff,
     'cores': cores
    })

df = paralell_eff.pivot(index='cores', columns='sizes', values='par_eff')


df.plot()
plt.xticks(cores)
plt.ylabel("Parallel eff")
plt.show()


# EXPERIMENT 2

mean_runtime_2 = [31.1961, 0.8708, 0.9592, 3.4610, 8.3066]
patch_sizes = [1,10,30,200,500]
plt.plot(patch_sizes, mean_runtime_2)
plt.xticks(patch_sizes)
plt.ylabel("Run time (s)")
plt.xlabel("Patch size")
plt.show()

# EXPERIMENT 3
c = []
f = open("results3.dat", "r")
a=f.readlines()
for el in a:
    c.append(el.strip().split(";")[3])

means = []
res = []
counter = 0
data = np.array(c, dtype=np.float32)
for index,el in enumerate(data):
    if counter == 3:
        res.append(np.mean(means))
        counter = 0
        means = []
    means.append(el)
    counter+=1

res.append(1.0326)
patch_sizes_exp3 = [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30]
print(len(res))
print(len(patch_sizes_exp3))
plt.plot(patch_sizes_exp3,res)
plt.ylabel("Run time (s)")
plt.xlabel("Patch size")
plt.show()
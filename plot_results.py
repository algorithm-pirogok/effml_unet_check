import json

import numpy as np
import matplotlib.pyplot as plt


with open("unet_benchmarks.json", "r", encoding="utf-8") as f:
    loaded = json.load(f)

x = np.array([[json.loads(k)['batch_size'], v['avg_time_per_image_ms']] for k, v in loaded.items() if v['method'] == "normal_fp16"])
y = x[:, 1]
x = x[:, 0]
plt.plot(x, y, label='normal_fp16')

x = np.array([[json.loads(k)['batch_size'], v['avg_time_per_image_ms']] for k, v in loaded.items() if v['method'] != "normal_fp16"])
y = x[:, 1]
x = x[:, 0]
plt.plot(x, y, label='torch_compile')

plt.legend()
plt.xlabel('batch size')
plt.ylabel('avg time per image (ms)')
plt.savefig("benchmark_results.png", dpi=150, bbox_inches="tight")
plt.show()

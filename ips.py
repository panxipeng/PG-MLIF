import time
import torch
from torch.nn.parameter import Parameter
import numpy as np
import torch
import time

# 加载已训练好的模型
# model = torch.load('lmf_test_model.pth')
model = torch.load('lmf_trained_model.pth')
# model = torch.load('tfn_trained_model.pth')
model.eval()

# 准备输入数据
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# vec1 = np.random.randint(0, 10, (32, 32))
vec1 = np.random.randn(32, 32)
vec1 = torch.tensor(vec1,dtype=torch.float32).to(device)
# vec2 = np.random.randint(0, 10, (32, 320))
vec2 = np.random.randn(32, 320)
vec2 = torch.tensor(vec2,dtype=torch.float32).to(device)
# input_data = torch.randn(16, 3, 224, 224)  # 示例输入，大小为(batch_size, channels, height, width)

# 进行模型推断并测量推断次数
num_inference = 1000  # 假设进行100次推断
total_time = 0

for _ in range(num_inference):
    start_time = time.time()
with torch.no_grad():
    _, output = model(x_path=vec1.to(device), x_grph=0, x_omic=vec2.to(device))
    # _, output = model(vec1,0,vec2)
end_time = time.time()

inference_time = end_time - start_time
total_time += inference_time

average_inference_time = total_time / num_inference
inferences_per_second = 1 / average_inference_time

print(f'平均推断时间：{average_inference_time:.5f} 秒')
print(f'每秒推断次数：{inferences_per_second:.2f}')

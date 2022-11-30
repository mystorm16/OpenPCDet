import torch
import time

batch_dict = torch.load("/media/ar304/T7/OpenPCDet/tools/test_batch.pth")
model = torch.load("/media/ar304/T7/OpenPCDet/tools/saved_model.pth")
for i in range(10):
    t1 = time.perf_counter()
    batch_dict = model(batch_dict)
    t2 = time.perf_counter()
    print(t2 - t1)

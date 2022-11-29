import torch
import time

batch_dict = torch.load("/media/ar304/T7/OpenPCDet/tools/test_batch.pth")
module_list = torch.load("/media/ar304/T7/OpenPCDet/tools/bev_shape_model.pth")
model = torch.nn.Sequential(module_list[0],
                            module_list[1],
                            module_list[2],
                            module_list[3])
for i in range(10):
    t1 = time.perf_counter()
    batch_dict = model(batch_dict)
    t2 = time.perf_counter()
    print(t2 - t1)
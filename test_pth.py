import torch
import time

batch_dict = torch.load("/media/ar304/T7/OpenPCDet/tools/test_batch.pth")
module_list = torch.load("/media/ar304/T7/OpenPCDet/tools/bev_shape_model.pth")
for i in range(10):
    t1 = time.perf_counter()
    for cur_module in module_list:
        batch_dict = cur_module(batch_dict)
    t2 = time.perf_counter()
    print(t2 - t1)

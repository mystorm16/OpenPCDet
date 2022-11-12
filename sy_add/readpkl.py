import pickle

file = open("/media/ar304/T7/备份11.4/OpenPCDet/output/kitti_models"
            "/voxel_rcnn_car_centerhead/default/eval/epoch_120/val/default/result.pkl", "rb")
data = pickle.load(file)
print(data)
file.close()

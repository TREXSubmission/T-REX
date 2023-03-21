import cv2
import os
num=0
dir = '/home/bovey/Downloads/Identity/logs/mobilenet_Pvoc_mc_full/2022-12-21-17-05-59/photos/LayerCAM'
print(os.listdir(dir))
for i in os.listdir(dir):
    new_dir = os.path.join(dir,i)
    for new in os.listdir(new_dir):
        last_dir = os.path.join(new_dir,new)
        name = '0.jph'
        for w in os.listdir(last_dir):
            try:
                if int(name.split('.')[0]) < int(w.split('.')[0]):
                    name = w
            except:
                pass
        photo = cv2.imread(os.path.join(last_dir,name))
        if photo[:,:,0].std() == photo[:,:,1].std() == photo[:,:,2].std():
            num+=1
            print(last_dir)
print(num)
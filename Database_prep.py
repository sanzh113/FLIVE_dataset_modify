import os
import shutil
import glob
import cv2
import pandas as pd
from tqdm import tqdm


#Extract VOC Train Validation tar File, Rename Images, and Move Image Files
#解压下载的VOC2012 trainval到database/voc_emotic_ava下
files = "database/voc_emotic_ava/VOCdevkit/VOC2012/JPEGImages/*"
with tqdm(total = int(len(glob.glob(files))), position=0, leave=True) as pbar:
    for path in glob.glob(files):
        name = path.split("\\")[-1]#由于window路径问题，用/这里获得的是文件夹的名字，换成了\\
        #print("name:",name)
        newp1 = path.replace(name,"")+"JPEGImages__"+name
        newp2 = path.replace(name, "")+"VOC2012__"+name
        os.rename(path, newp1)
        shutil.copy(newp1, 'database/voc_emotic_ava/')
        os.rename(newp1, newp2)
        shutil.copy(newp2, 'database/voc_emotic_ava/')
        os.remove(newp2)
        pbar.update()
    pbar.update()
    pbar.close()
shutil.rmtree('database/voc_emotic_ava/VOCdevkit')#删除文件
print("VOC - Done!")


### Extract VOC Test tar File, Rename Images, and Move Image Files
files = "database/voc_emotic_ava/VOC2012test/VOCdevkit/VOC2012/JPEGImages/*"
with tqdm(total = int(len(glob.glob(files))), position=0, leave=True) as pbar:
    for path in glob.glob(files):
        name = path.split("\\")[-1]
        newp1 = path.replace(name,"")+"JPEGImages__"+name
        newp2 = path.replace(name, "")+"VOC2012__"+name
        os.rename(path, newp1)
        shutil.copy(newp1, 'database/voc_emotic_ava/')
        os.rename(newp1, newp2)
        shutil.copy(newp2, 'database/voc_emotic_ava/')
        os.remove(newp2)
        pbar.update()
    pbar.update()
    pbar.close()
shutil.rmtree('database/voc_emotic_ava/VOC2012test')
print("VOC_test_set_2012 - Done!")




### Extract Emotic Zip File, Rename Images, and Move Image Files
#emotic.zip'解压到database/voc_emotic_ava/emotic
folders = os.listdir("database/voc_emotic_ava/emotic/")
with tqdm(total = len(folders), position=1, leave=True) as pbar:
    for folder in folders:
        files = "database/voc_emotic_ava/emotic/"+folder + "/images/*"
        for path in glob.glob(files):
            name = path.split("\\")[-1]
            newp = path.replace(name,"")+"EMOTIC__"+name
            os.rename(path, newp)
            shutil.copy(newp, 'database/voc_emotic_ava/')
            os.remove(newp)
        pbar.update()
    pbar.close()
shutil.rmtree('database/voc_emotic_ava/emotic')
shutil.rmtree('database/voc_emotic_ava/__MACOSX')
print("Emotic - Done!")

#Rename and Move AVA Images
if os.path.isdir('database/AVA_dataset') == True:
    files = 'database/AVA_dataset/image/*'#images改成了image，下载的数据集里是image
    for path in tqdm(glob.glob(files)):
        name = path.split("\\")[-1]
        #print(name)
        newp = path.replace(name, "")+"AVA__"+name
        os.rename(path, newp)
        shutil.move(newp, 'database/voc_emotic_ava/')
    #shutil.rmtree('database/voc_emotic_ava/AVA_dataset')
    print("AVA - Done!")

#删掉不需要的图片
df_ims = pd.read_csv('labels_image.csv')
ims = df_ims['name']
ims = ims.values.tolist()
#   这里对blur dataset和voc_emotic_ava文件夹操作就行
folders = ['blur_dataset','voc_emotic_ava']
#folders = os.listdir('database')

for folder in folders:
    files = 'database/'+folder+"/*"
    for path in glob.glob(files):
        name = path.replace("database/","")
        #这里的\\要变成/
        name = name.replace("\\", "/")
        if name not in ims:
            os.remove(path)
print("Cleaning - Done!")

#分块
try:
    if os.path.isdir('database/patches') == False:
        os.mkdir('database/patches')
except OSError:
    print ("Creation of the directory %s failed" % 'patch')

def patch_sampling(im_path, patch_index, x, y, width, height):
    img=cv2.imread(im_path)
    patch = img[y:y + height, x:x + width]
    name = im_path.split('\\')[-1]
    p_path = 'database/patches/'+name+"_"+"patch_"+str(patch_index)+".jpg"
    cv2.imwrite(p_path,patch)

df_coor = pd.read_csv('all_patches.csv')
folders = ['blur_dataset','voc_emotic_ava']

for folder in folders:
    if folder != 'patches':
        im_paths = 'database/'+folder+"/*"
        for path in glob.glob(im_paths):
            print("path:",path)
            im_name = path.split('\\')[-1]#读取图片名
            patch_name = "patches/"+im_name
            row = df_coor.loc[df_coor['name_patch']==patch_name]#在csv文件中找到指定的图片那行数据
            if len(row) == 1:
                #x1 = int(row['top_patch_1'].values.tolist()[0])
                x1 = int(row['left_patch_1'].values.tolist()[0])
                #y1 = int(row['left_patch_1'].values.tolist()[0])
                y1 = int(row['top_patch_1'].values.tolist()[0])

                x2 = int(row['left_patch_2'].values.tolist()[0])
                y2 = int(row['top_patch_2'].values.tolist()[0])
                x3 = int(row['left_patch_3'].values.tolist()[0])
                y3 = int(row['top_patch_3'].values.tolist()[0])
                h1 = int(row['height_patch_1'].values.tolist()[0])
                w1 = int(row['width_patch_1'].values.tolist()[0])
                h2 = int(row['height_patch_2'].values.tolist()[0])
                w2 = int(row['width_patch_2'].values.tolist()[0])
                h3 = int(row['height_patch_3'].values.tolist()[0])
                w3 = int(row['width_patch_3'].values.tolist()[0])
                patch_sampling(path, 1, x1, y1, w1, h1)#分割图片
                patch_sampling(path, 2, x2, y2, w2, h2)
                patch_sampling(path, 3, x3, y3, w3, h3)
print("Patch sampling - Done!")
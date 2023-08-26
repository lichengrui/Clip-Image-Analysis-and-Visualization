import torch
import clip
from PIL import Image
import os
import numpy as np
from tqdm import tqdm
import time
import json 
from sklearn.cluster import KMeans
# from sklearn.preprocessing import StandardScaler
from random import randint
import math
# import matplotlib.pyplot as plt

class ClipThing:

  def __init__(self):
    self.device = "cuda" if torch.cuda.is_available() else "cpu"
    self.model, self.preprocess = clip.load("ViT-B/32", device=self.device)
  
  def create_dir(self,folder = "IMAGES"):
    print("STARTING INSANITY!")
    progress = 0
    fs = 0
    start = time.time()
    for f in os.scandir(folder):
        if f.is_dir():
            y = []
            image_to_label = []
            for image_name in tqdm(os.scandir(f)):
                p = folder + "/" + f.name + "/" + image_name.name
                image = self.preprocess(Image.open(p)).unsqueeze(0).to(self.device)
                with torch.no_grad():
                    imf = self.model.encode_image(image)
                    y.append(imf.detach().cpu().numpy().reshape(512))
                    image_to_label.append([p,progress])
                progress += 1
            elapsed_time = (time.time() - start)/3600
            estimated_remaining_time = (elapsed_time*1300000 / progress-elapsed_time)
            print("Estimated Remaining Time (Hrs): "  + str(estimated_remaining_time) + ",  Total Elapsed Time (Hrs):"+ str(elapsed_time))
            np.save(str(r"npyfiles\\" + f.name + ".npy"), y)
            np.save(str(r"npylabels\\" + f.name + "_LABELS.npy"), image_to_label)

  def combine_img_dir(self, filename = "all_images", image_folder = 'npyfiles'):
    q = True
    for a in tqdm(os.scandir(image_folder)):
      temp = np.load(a)
      if q:
        images = temp
        q = False
      else:
        images = np.concatenate((images,temp), axis= 0)
    np.save(filename+".npy",images)

  def combine_lab_dir(self, filename = "all_labels", label_folder = 'npylabels'):
    l = True
    for b in tqdm(os.scandir(label_folder)):
      temp = np.load(b)
      if l:
        labels = temp
        l = False
      else:
        labels = np.concatenate((labels,temp), axis= 0)
    np.save(filename +".npy",labels)

  def read_dir(self, img_filename = "all_images.npy", lab_filename = "all_labels.npy"):
    self.images = np.load(img_filename)
    print("Images Loaded")
    self.labels = np.load(lab_filename)
    print("Labels Loaded")

  def searchtext(self,query,k=50):
    print("Searching for: " + query)

    #Convert Text to Features
    text = clip.tokenize(query).to(self.device)
    with torch.no_grad():
        text_feature = self.model.encode_text(text)
    # print(type(text_feature))
    

    #Compute Similarities
    similarity = text_feature.cpu().numpy() @ self.images.T
    # print(similarity.shape)
    similarity = torch.from_numpy(similarity)
    similarity = similarity.float()
    similarity = similarity.softmax(dim=-1)
    similarity = similarity.cpu().numpy()

    #Create Image List
    imagethings = self.labels[similarity.argsort()]
    sorted_similarity = similarity.argsort()
    imagethings = np.flip(imagethings)
    sorted_similarity = np.flip(sorted_similarity)
    imagethings = imagethings.reshape((imagethings.shape[1],imagethings.shape[2]))
    top_images = imagethings[:k]
    temp = self.images[sorted_similarity.T[:k]]
    temp = temp.reshape((k,512))
    print(temp.shape)
    # print(top_images)
    # print(similarity.shape)
    top_fp = []

    #Creating JSON
    print("Found Images")
    imglink = "searchthing.png"
    out_js = {"name": query, "img" : imglink,"children":[],"sim":1}
    tempsim = []
    IDlist = []

    count = 0
    #Normalizing Similarities
    for i in top_images:
      tempsim.append(float(similarity[:,int(i[0])]))
      IDlist.append({"ID":int(i[0]),"img":i[1],"count": count})
      count += 1
    tempsim = np.array(tempsim)
    print(tempsim)
    norm_sim = (tempsim - np.amin(tempsim))/np.amax(tempsim - np.amin(tempsim))
    print(norm_sim)
    tempsim = np.array(tempsim)
    asd = np.log(tempsim)
    asd = asd - np.amin(asd)
    asd[0] = 0
    # print(asd)
    asd = (asd - np.amin(asd))/np.amax(asd)
    norm_sim = asd

    print('CLUSTERING')
    kmeans = KMeans(n_clusters=int(k/randint(7,15)), random_state=0).fit(temp)
    lab =  kmeans.labels_

    #Creating JSON Structure
    for q in range(max(lab)+1):
      tf = True
      for num, i in enumerate(top_images):
        if q == lab[num]:
          if tf:
            child = {"sim": float(norm_sim[num]),"ID": str(i[0]),"img":str(i[1]),"size": 40000,"children":[]}
            tf = False
          else:
            top_fp.append(i[1])
            new = {"sim": float(norm_sim[num]),"ID": str(i[0]),"img":str(i[1]),"size": 40000}
            child['children'].append(new)
      out_js["children"].append(child)

    return(top_fp,out_js,IDlist)

  def searchID(self,ID,k=50):
    print("Searching for ID: " + ID)

    #Convert Text to Features
    ID_feature = self.images[int(ID)]

    #Compute Similarities
    similarity = ID_feature @ self.images.T
    similarity = similarity.reshape((1,similarity.shape[0]))
    # print("ID")
    # print(similarity.shape)
    similarity = torch.from_numpy(similarity)
    similarity = similarity.float()
    similarity = similarity.softmax(dim=-1)
    similarity = similarity.cpu().numpy()

    #Create Image List
    imagethings = self.labels[similarity.argsort()]
    sorted_similarity = similarity.argsort()
    imagethings = np.flip(imagethings)
    sorted_similarity = np.flip(sorted_similarity)
    imagethings = imagethings.reshape((imagethings.shape[1],imagethings.shape[2]))
    top_images = imagethings[:k]
    temp = self.images[sorted_similarity.T[:k]]
    temp = temp.reshape((k,512))
    # print(top_images)
    # print(similarity.shape)
    top_fp = []

    #Creating JSON
    print("Found Images")
    imglink = self.labels[int(ID)][0]
    out_js = {"name": ID, "img" : imglink,"children":[],"sim":1}
    tempsim = []
    IDlist = []

    #Normalizing Similarities
    count = 0
    for i in top_images:
      tempsim.append(float(similarity[:,int(i[0])]))
      IDlist.append({"ID":int(i[0]),"img":i[1],"count": count})
      count += 1
    tempsim = np.array(tempsim)
    asd = np.log(tempsim)
    asd = asd - np.amin(asd)
    asd[0] = 0
    # print(asd)
    asd = (asd - np.amin(asd))/np.amax(asd)
    norm_sim = asd


    print('CLUSTERING')
    kmeans = KMeans(n_clusters=int(k/randint(4,8)), random_state=0).fit(temp)
    lab =  kmeans.labels_

    #Creating JSON Structure
    # for num, i in enumerate(top_images):
    #   if str(i[1]) == imglink:
    #     continue
    #   top_fp.append(i[1])
    #   new = {"sim": float(norm_sim[num]),"ID": str(i[0]),"img":str(i[1]),"size": 40000}
    #   out_js["children"].append(new)
    for q in range(max(lab)+1):
      tf = True
      for num, i in enumerate(top_images):
        if q == lab[num] and str(i[1]) != imglink:
          if tf:
            child = {"sim": float(norm_sim[num]),"ID": str(i[0]),"img":str(i[1]),"size": 40000,"children":[]}
            tf = False
          else:
            top_fp.append(i[1])
            new = {"sim": float(norm_sim[num]),"ID": str(i[0]),"img":str(i[1]),"size": 40000}
            child['children'].append(new)
      out_js["children"].append(child)

    return(top_fp,out_js,IDlist)

  def writejson(self, data,filename = "image.json"):
    # print(data)
    with open(filename, "w") as outfile:
      json.dump(data, outfile)


if __name__ == '__main__':
  query = "Dog"
  folderpath = "D:/CLIP PROJECT THING/"

  c = ClipThing()
  c.read_dir()
  files,out_js,IDs = c.searchtext(query)
  path = folderpath + "static/image.json"
  c.writejson(out_js,path)
  path = folderpath + "static/smolimage.json"
  c.writejson(IDs,path)
  #print(files)
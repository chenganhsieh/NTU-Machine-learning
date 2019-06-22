import numpy as np
from PIL import Image
import torch
from torch.autograd import Variable
from torch.autograd.gradcheck import zero_gradients
import torch.nn as nn 
import torch.nn.functional as F 
import torchvision 
import torch.optim as optim 
from torchvision import transforms
from tqdm import *
import torchvision.models 
import csv
from sys import argv
imageSize = (224, 224)

def mytransform():
    transform1= transforms.Compose([
        transforms.Resize(imageSize),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
    return transform1   
def image_location_generator(_root):
    import os
    _imgfiles= os.listdir(_root)
    assert len(_imgfiles) > 0, "Empty!!"
    for _imgfile in _imgfiles:
        yield os.path.join(_imgfile)
def load_image(imageRaw):
    image = Image.open(imageRaw)
    image = image.convert("RGB") 
    image = mytransform()(image).float()
    image = image.unsqueeze(0)  
    return image 
def readCategories(filename):
    file = open(filename, 'r')
    mapping = []
    n_row = 0
    for row in csv.reader(file):
        if n_row > 0:
            mapping.append(row[1])            
        n_row += 1
    return mapping
def readLabel(filename):
    file = open(filename, 'r')
    mapping = []
    n_row = 0
    for row in csv.reader(file):
        if n_row > 0:
            mapping.append(float(row[3]))            
        n_row += 1
    return mapping
def save_image(img, path):
    im = img[0] 
    im[0] = im[0] * 0.229
    im[1] = im[1] * 0.224 
    im[2] = im[2] * 0.225 
    im[0] += 0.485 
    im[1] += 0.456 
    im[2] += 0.406
    im = np.moveaxis(im, 0, 2)
    im=(im*255).astype(np.uint8)
    im = Image.fromarray(im)
    im.save(path)


def main():
    resnet50 = torchvision.models.resnet50(pretrained=True) 
    resnet50= resnet50.cuda()
    resnet50.eval() 
    Loss_fn = nn.CrossEntropyLoss()
    print (".. loaded  resnet50")

    imNum=0
    succAttack=0
    Linfinity=0.0
    imageLocation=""
    saveLocation=""
    if argv[1][-1]!="/":
        imageLocation=argv[1]+"/"
    else:
        imageLocation=argv[1]
    if argv[2][-1]!="/":
        saveLocation=argv[2]+"/"
    else:
        saveLocation=argv[2]
    for imloc in tqdm(image_location_generator(imageLocation),total=200):
        if imloc.split(".")[1]=="png":
            numImage=int(float(imloc.split(".")[0]))
        else:
            continue
        imNum+=1

        x = Variable(load_image(imageLocation+imloc).cuda(), requires_grad=True)
        x_test=Variable(load_image(imageLocation+imloc).cuda(), requires_grad=True)
        
        adv_x=x.clone()
        netOriginIndex=int(np.argmax(resnet50.forward(x).detach().cpu().numpy()))
        zero_gradients(adv_x)
        output=resnet50.forward(adv_x)
        lossCal=Loss_fn(output,Variable(torch.LongTensor(np.array([output.detach().cpu().numpy().argmax()])), requires_grad = False).cuda())
        lossCal.backward()
        x_grad=0.007*torch.sign(x.grad.data)
        adv_x.data=adv_x.data+x_grad
                
        
        if int(np.argmax(resnet50.forward(adv_x).detach().cpu().numpy()))==netOriginIndex: 
            adv_x=x.data
            tqdm.write(imloc)
                 
        
        
        
        y_pred_adversarial = int(np.argmax(resnet50.forward(adv_x).detach().cpu().numpy()))
        y_true = netOriginIndex
       
        if y_pred_adversarial == y_true:
            tqdm.write("Adversarialize FAIL! ")
        else:
            succAttack+=1
            
        save_image(adv_x.detach().cpu().numpy(),saveLocation+imloc)


        result=adv_x.detach().cpu().numpy()[0]
        myresult=np.concatenate((result[0],result[1],result[2]),axis=0)


        origin=x_test.detach().cpu().numpy()[0]
        originResult=np.concatenate((origin[0],origin[1],origin[2]),axis=0)
        
        Linfinity+=abs(np.linalg.norm(myresult,np.inf)-np.linalg.norm(originResult,np.inf))
        tqdm.write(str(Linfinity))
        #save_image(adv_x.detach().cpu().numpy(),"./outputCheat/"+imloc)
    print("SuccessRate:"+str(float(succAttack/imNum)*100)+"%")        
    #print("L-infinity"+str(float(Linfinity)))    
    
if __name__ == '__main__':
    main()

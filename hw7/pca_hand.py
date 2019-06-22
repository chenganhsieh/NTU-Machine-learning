
import numpy as np
from sys import argv
from skimage import io
import os
def loadImage(directory):
	image = []
	for i in range(415):
		file_name = directory + '/' + str(i) + '.jpg'
		img = io.imread(file_name)
		img = np.array(img)
		image.append(img.flatten())
	image = np.array(image)
	return image

def reproduce(val,val_mean,u,top):
	val = val - val_mean
	weight = val.T.dot(u[:,:top])	
	img = u[:,:top].dot(weight.T)
	img = img + val_mean 	
	return img
def saveImage(image,name):
	image -= np.min(image)
	image /= np.max(image)

	image = (image*255).astype(np.uint8).reshape(600,600,3)
	io.imsave(name, image)

def main():
	mean_image=[]
	print("==Load Image==")
	data=loadImage(argv[1])
	data = data.T

	print("==Caculating...==")
	data_mean = data.mean(axis=1).reshape(-1,1)
	#u,s,v = np.linalg.svd(data_mean, full_matrices = False)
	#u = u*(-1) 
	#rep_img = reproduce(data_mean,data_mean,u,5)
	#saveImage(rep_img, "./result/pervage_0.jpg")
	#print("===Done===")
	#input()
	data = data - data_mean
	
	u,s,v = np.linalg.svd(data, full_matrices = False)
	u = u*(-1) 
	#for i in range(5):
	#	saveImage(u[:,i],"test_"+str(i)+".jpg")	
	##saveImage(u[:,10],"test_"+str(10)+".jpg")	
	#print("Done")
	#input()
	#print(s)
	#eigenValue=np.square(s)/414
	#total=np.sum(eigenValue)
	#s=0
	#for i in eigenValue:
	#	s+=1
	#	print(i/total*100)
	#	if s>6:
	#		break
	
	#seigenValue=eigenValue.sort()

	#u = u*(-1)   #be positive
	temp=io.imread(os.path.join(argv[1],argv[2]))
	ori_image=temp.flatten().T.reshape(-1,1)
	rep_img = reproduce(ori_image,data_mean,u,5)
	saveImage(rep_img, argv[3])
	print("===Done===")


	

if __name__ == '__main__':
	main()
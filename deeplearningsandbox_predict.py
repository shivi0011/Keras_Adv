import sys, os
import argparse
import numpy as np
import cv2
from PIL import Image, ImageDraw, ImageFont
import requests
from io import BytesIO
import matplotlib.pyplot as plt

#from keras.preprocessing import image
from keras.models import load_model
from keras.applications.inception_v3 import preprocess_input


target_size = (229, 229) #fixed size for InceptionV3 architecture
#target_size = (256, 256) #fixed size for InceptionV3 architecture

#LABEL = ("airplane","cloud","fields","forest","urban","waterbody")
LABEL = ("fields","urban")

def predict(model, img, target_size):
  """Run model prediction on image
  Args:
    model: keras model
    img: PIL format image
    target_size: (w,h) tuple
  Returns:
    list of predicted labels and their probabilities
  """
  ''' this will work when u will use- img = Image.open(args.image)
  if img.size != target_size:
    img = img.resize(target_size)
  '''

  
  imgg = cv2.imread(img)

  gray = cv2.cvtColor(imgg, cv2.COLOR_BGR2GRAY)     
  x = np.zeros_like(imgg)
  x[:,:,0] = gray
  x[:,:,1] = gray
  x[:,:,2] = gray
  #cv2.imshow(x)
  x = cv2.resize(x,(256,256))
  x = np.array(x)
  x = x.astype('float32')
  x /= 255
  x = np.expand_dims(x, axis=0)  
  print (x.shape)

  x = preprocess_input(x)
  preds = model.predict(x)
  return preds[0]


def plot_preds(image, preds):
  """Displays image and the top-n predicted probabilities in a bar graph
  Args:
    image: PIL image
    preds: list of predicted labels and their probabilities
  """
  #plt.imshow(image)
  #plt.axis('off')

  plt.figure()
  labels = LABEL
  plt.barh([0, 1,2,3,4,5], preds, alpha=0.5)
  plt.yticks([0, 1,2,3,4,5], labels)
  plt.xlabel('Probability')
  plt.xlim(0,1.01)
  plt.tight_layout()
  plt.show()


# =============================================================================
# def show_multiple_preds(model, test_img_dir):
#   #makeList = "ls -tr "+test_img_dir+"/*_*_*.jpg > "+"C:/Users/Adrin/Desktop/keras_bin/GoogleNet_exp/deeplearning_sandbox/test_img_dir_resized/inference.txt"
#   #print(makeList)
#   #os.system(makeList)
#   path1 = "C:/Users/Adrin/Desktop/keras_bin/GoogleNet_exp/deeplearning_sandbox/test_img_dir"    #path of folder of images    
#   path2 = 'C:\\Users\\Adrin\\Desktop\\keras_bin\\GoogleNet_exp\\deeplearning_sandbox\\test_img_dir_resized'  #path of folder to save images    
#   listing = os.listdir(path1)
#   
# # =============================================================================
# #   for file in listing:
# #       im = Image.open(path1 + "/" + file)  
# #       img = im.resize((256, 256))
# #       gray = img.convert('L')
# #                 #need to do some more processing here          
# #       gray.save(path2 +'\\' +  file, 'JPEG')
# # =============================================================================
#   
#   #image=Image.open(sys.argv[1])
#   image=Image.open(path1+"/000069_52.jpg")
#   #tmpfile=
#   tmpfp = open(path2+'\\inference.txt',"r")
#   draw=ImageDraw.Draw(image)
#   for line in tmpfp:
#     tilename = os.path.split(line)[1]
#     img = Image.open(line.split("\n")[0])
#     preds = predict(model, img, target_size)
#     maxItem =  max(preds)
#     maxItemIndex = np.where(preds==maxItem)
#     print(LABEL[maxItemIndex[0][0]])
#     labeltext = LABEL[maxItemIndex[0][0]]
#     r = tilename.split('_')[-2]
#     c = tilename.split('_')[-1].split('.')[0]	
#     label_color="cyan"
#     #font_type=ImageFont.truetype("/usr/share/fonts/truetype/ubuntu-font-family/UbuntuMono-B.ttf",25)
#     draw.text(xy=(int(c)+128,int(r)+128),text=labeltext,fill=label_color)
#   image.save(test_img_dir+"/"+"inference.jpg")    
#   image.show()
# =============================================================================


''' Command to run
python deeplearningsandbox_predict.py --image fields1.jpg --model inceptionv3-ft_multiclass.model

for multiple image
python deeplearningsandbox_predict.py  --model inceptionv3-ft_multiclass.model
'''
if __name__=="__main__":
  a = argparse.ArgumentParser()
  a.add_argument("--image", help="path to image")
  a.add_argument("--image_url", help="url to image")
  a.add_argument("--model")
  a.add_argument("--test_img_dir", default="C:/Users/Adrin/Desktop/test_img_dir")
  args = a.parse_args()

  if args.image is None and args.image_url is None:
    a.print_help()
    sys.exit(1)

  model = load_model(args.model)
  if args.image is not None:
    #img = Image.open(args.image)
    img = args.image
    preds = predict(model, img, target_size)
    print("Prediction for image "+args.image+" : ",preds)
    img = Image.open(args.image)
    #img.show()
    #plt.imshow(img)
    plot_preds(img, preds)

# =============================================================================
#   if args.test_img_dir is not None:
#     #img = Image.open(args.image)
#     #preds = predict(model, img, target_size)
#     #print("Prediction for image "+args.image+" : ",preds)
#     test_img_dir = args.test_img_dir
#     print(test_img_dir)
#     #show_multiple_preds(model, test_img_dir)
# =============================================================================


  if args.image_url is not None:
    response = requests.get(args.image_url)
    img = Image.open(BytesIO(response.content))
    preds = predict(model, img, target_size)
    plot_preds(img, preds)


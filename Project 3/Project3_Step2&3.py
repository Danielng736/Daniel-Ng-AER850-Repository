from ultralytics import YOLO
from PIL import Image
from google.colab import drive
drive.mount('/content/drive')

model = YOLO('yolov8n.pt')  

results = model.train(data='/content/drive/MyDrive/Project3Data/data/data.yaml', epochs=190, imgsz=900, batch=20)

##################################################################################

model = YOLO('/content/runs/detect/train/weights/best.pt')

# Validate the model
metrics = model.val()  
metrics.box.map    
metrics.box.map50  
metrics.box.map75  
metrics.box.maps   

# Run batched inference on a list of images
results = model(['/content/drive/MyDrive/Project3Data/data/evaluation/rasppi.jpg'])

# Process results list
for result in results:
    boxes = result.boxes 
    masks = result.masks  
    keypoints = result.keypoints  
    probs = result.probs  

# Show the results
for r in results:
    im_array = r.plot(line_width=3, font_size=3)  
    im = Image.fromarray(im_array[..., ::-1])  
    im.show()  
    im.save('/content/drive/MyDrive/resultsrasppi.jpg')  
    
##################################################################################

model = YOLO('/content/runs/detect/train/weights/best.pt')

# Validate the model
metrics = model.val() 
metrics.box.map    
metrics.box.map50  
metrics.box.map75  
metrics.box.maps   

# Run batched inference on a list of images
results = model(['/content/drive/MyDrive/Project3Data/data/evaluation/ardmega.jpg'])

# Process results list
for result in results:
    boxes = result.boxes  
    masks = result.masks  
    keypoints = result.keypoints 
    probs = result.probs  

# Show the results
for r in results:
    im_array = r.plot(line_width=3, font_size=3)  
    im = Image.fromarray(im_array[..., ::-1])
    im.show()  # show image
    im.save('/content/drive/MyDrive/resultsardmega.jpg')  
    
##################################################################################

model = YOLO('/content/runs/detect/train/weights/best.pt')

# Validate the model
metrics = model.val()  
metrics.box.map    
metrics.box.map50  
metrics.box.map75  
metrics.box.maps   

# Run batched inference on a list of images
results = model(['/content/drive/MyDrive/Project3Data/data/evaluation/arduno.jpg'])

# Process results list
for result in results:
    boxes = result.boxes  
    masks = result.masks  
    keypoints = result.keypoints  
    probs = result.probs  

# Show the results
for r in results:
    im_array = r.plot(line_width=2, font_size=2)  
    im = Image.fromarray(im_array[..., ::-1])  
    im.show()  
    im.save('/content/drive/MyDrive/resultsarduno.jpg')  
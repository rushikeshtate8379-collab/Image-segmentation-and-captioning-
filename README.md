# Image-segmentation-and-captioning-
A Deep Learning project that can describe images with captions and segment objects in them—perfect for AI-powered image understanding!

⚡ Features

Image Captioning: Generate descriptive captions using a Transformer-based model.
Image Segmentation: Segment objects with U-Net, Mask R-CNN, or a ResNet50-based model.
Interactive Web App: Upload images and instantly get captions & segmentation masks.
Object Detection: ResNet50 version draws bounding boxes and individual masks for each object.
Pretrained Models: Ready-to-use weights for quick inference or optional retraining.

Install deppendencies: This project requries Python 3.8+ and the following packagess:

streamlit
tensorflow
torch
torchvision
numpuy
pandas
pillow
matplotlib
opencv-python
request
tqdm
scikit-image(if used in notebooks)

Install them with:

pip install streamlit tensorflow torch torchvision numpy pandas pillow matplotlib opencv-python requests tqdm

​📝 Guide to Use:
1. ​Run Image_Caption_train.ipynb on kaggle
​2. Save the output model in a folder named "saved_models" as "image_captioning_coco_weights" & output "vocab_coco.file" in a folder named "saved_vocabulary"
​3. Run image-segmentation.ipynb on kaggle
​4. Save this model in saved_models folder as "segmentation_model"
​5. Run app.py using streamlit
​
NOTE
​I have also made a version of this which uses pretrained RESNET50 for image segmentation. It also marks objects in a box and gives indivisual masks for each object.

​Guide to Use:
1. ​Run Image_Caption_train.ipynb on kaggle
​2.Save the output model in a folder named "saved_models" as "image_captioning_coco_weights" & output "vocab_coco.file" in a folder named "saved_vocabulary"
​3. Run Pretrain_app.py using streamlit
​

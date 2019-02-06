# COFGA: Classification Of Fine-Grained Features In Aerial Images  
```
pip install -upgrade pip  
pip install -r requirments.txt  
```  

downloading pretrained was not working for some reason. Mannualy downlod model  
```  
wget https://download.pytorch.org/models/resnet18-5c106cde.pth                                 
mv resnet18-5c106cde.pth  ~/.torch/models/  
```  
# visualization
use display_class and display_test. the differences are that test doesn't have labels

# prepare split
The train set (as it turns out) contains overlapping strips. Hence the same instance can appear in multiple images. Use train_split.py to detect overlaping image groups (or use data/train_image_groups.yaml that was computed with train_split.py). Use the yaml file to initialize the data_loaders.

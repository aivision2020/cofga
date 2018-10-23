pip install -upgrade pip
pip install -r requirments.txt

#downloading pretrained was not working for some reason. Mannualy downlod model
wget https://download.pytorch.org/models/resnet18-5c106cde.pth                               
mv resnet18-5c106cde.pth  ~/.torch/models/

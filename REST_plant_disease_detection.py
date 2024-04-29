import warnings
warnings.filterwarnings('ignore')
import json
import torch
import torch.nn as nn
import torchvision.models as models
from PIL import Image
import matplotlib.pyplot as plt
import torch.nn.functional as F
import gc
import base64
from io import BytesIO
from flask import Flask, request, render_template, jsonify
import torchvision.transforms as transforms
from flask_cors import CORS



with open(r'C:\\Users\\User\\Downloads\\Plant_disease\\REST\\class_mapper.json', 'r') as json_file:
    class2index = json.load(json_file)

index2class = {v:k for k, v in class2index.items()}


vgg16 = models.vgg16(weights='DEFAULT')


for name, param in vgg16.named_parameters():
    if 'features.0' in name or 'features.2' in name:
        param.requires_grad = True
    else:
        param.requires_grad = False


vgg16.classifier[-1] = nn.Linear(in_features=vgg16.classifier[-1].in_features, out_features=len(index2class))

vgg16.load_state_dict(torch.load(r'C:\\Users\\User\\Downloads\\Plant_disease\\REST\\VGG16_transfer_learning_trained_weights.pth',
                                 map_location=torch.device('cpu')),
                                 strict=False)

vgg16.eval()



transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


app = Flask(__name__)
CORS(app)



@app.route('/upload', methods=['POST'])
def upload():
    file = request.files['image']
    image_stream = BytesIO(file.read())

    img = Image.open(image_stream)
    img_tensor = transform(img).unsqueeze(0)
    return jsonify({'image_tensor': img_tensor.tolist(), 'image_shape': img_tensor.shape})



@app.route('/predict', methods=['POST'])
def predict():
    request_data = request.json


    img_tensor = request_data['image_tensor']
    b, c, h, w = request_data['image_shape']
    img_tensor = torch.tensor(img_tensor, dtype=torch.float32).view(b, c, h, w)

    with torch.no_grad():
        out = vgg16(img_tensor)
    pred = torch.argmax(F.softmax(out, dim=-1), dim=-1)

    prediction = index2class[pred.item()]

    return jsonify({'prediction': prediction})
    



if __name__ == "__main__":
    # app.run(host='0.0.0.0', port=5000, debug=True)
    app.run(debug=True)


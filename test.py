import torch 
from torchvision import transforms
from PIL import Image 


model = torch.load('./fine_tuned_models/toxic_plant_model.pth')

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)

model.eval()


# Prepare the image for prediction 
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])



image_path = './test_data/test2.jpg'
image = Image.open(image_path)

# Apply transformations
image = transform(image).unsqueeze(0).to(device)

# Make a prediction
with torch.no_grad():
    outputs = model(image)
    _, preds = torch.max(outputs, 1)


class_names = ['non-toxic', 'toxic']
prediction_label = class_names[preds.item()]

print(f'Prediction: {prediction_label}')
import torch
import torch.nn as nn
from torchvision import models, transforms

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

model = models.resnet18(pretrained=False)
num_ftrs = model.fc.in_features
model.fc = nn.Linear(num_ftrs, 3)

model.load_state_dict(torch.load("./ml/model/v1-epoch-2.pth"))
model = model.to(device)

transform = transforms.Compose(
    [
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)

class_names = ["Govisetha", "Koti Kapruka", "Mahajana Sampatha"]


def predict_lottery_type(frame, model, transform, class_names):
    image = frame

    image = transform(image).unsqueeze(0)
    image = image.to(device)

    model.eval()

    with torch.no_grad():
        outputs = model(image)

        _, predicted = torch.max(outputs, 1)

        predicted_class = class_names[predicted.item()]

    return predicted_class


def get_lottery_type(image):
    return predict_lottery_type(image, model, transform, class_names)

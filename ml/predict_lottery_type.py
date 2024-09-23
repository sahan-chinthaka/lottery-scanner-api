import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image


def predict_lottery_type(frame, model, transform, class_names):
    # image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    image = frame

    image = transform(image).unsqueeze(0)
    image = image.to(device)

    model.eval()

    with torch.no_grad():
        outputs = model(image)
        print(outputs)

        _, predicted = torch.max(outputs, 1)

        predicted_class = class_names[predicted.item()]

    return predicted_class


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

model = models.resnet18(pretrained=False)
num_ftrs = model.fc.in_features
model.fc = nn.Linear(num_ftrs, 3)

model.load_state_dict(torch.load("./model/v1-epoch-2.pth"))
model = model.to(device)

transform = transforms.Compose(
    [
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)

class_names = ["Govisetha", "Koti Kapurka", "Mahajana Sampatha"]

predicted_class = predict_lottery_type(
    Image.open("./test.jpg"), model, transform, class_names
)

print(predicted_class)

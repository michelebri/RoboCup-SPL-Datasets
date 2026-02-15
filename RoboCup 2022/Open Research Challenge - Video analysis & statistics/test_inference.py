import torch
from torchvision import transforms
from PIL import Image

from color_classifier import ColorClassifierCNN

model_path = "color_classifier.pt"
image_path =  None # put your image here

checkpoint = torch.load(model_path, map_location="cuda" if torch.cuda.is_available() else "cpu")
classes = checkpoint["classes"]

model = ColorClassifierCNN(in_channels=3, num_classes=len(classes))
model.load_state_dict(checkpoint["model_state"])
model.eval()

transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
])

image = Image.open(image_path).convert("RGB")
x = transform(image).unsqueeze(0)  # add batch dim

with torch.no_grad():
    logits = model(x)
    probs = torch.softmax(logits, dim=1)
    pred_idx = probs.argmax(dim=1).item()
    conf = probs[0, pred_idx].item()

print("pred:", classes[pred_idx], "conf:", conf)
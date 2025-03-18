from fastapi import FastAPI, File, UploadFile
from torchvision import transforms, models
from PIL import Image
import torch
import torch.nn as nn
import io

app = FastAPI()

# --- Define the Model Class ---
class CatDogClassifier(nn.Module):
    def __init__(self):
        super(CatDogClassifier, self).__init__()
        self.model = models.resnet18(pretrained=True)
        self.model.fc = nn.Linear(512, 2)  # 2 classes (Cat/Dog)

    def forward(self, x):
        return self.model(x)

# --- Load the Trained Model ---
model = CatDogClassifier()  # Initialize model
model.load_state_dict(torch.load(r"E:\ml_assignment_tut\cat_vs_dog_state1.pth", map_location="cpu"))  # Load weights
model.eval()  # Set model to evaluation mode

# --- Define Image Transformations ---
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

# --- Class Labels ---
classes = ["Cat", "Dog"]

@app.post("/predict/")
async def predict(file: UploadFile = File(...)):
    try:
        # Read and preprocess image
        image = Image.open(io.BytesIO(await file.read())).convert("RGB")
        image = transform(image).unsqueeze(0)  # Add batch dimension
        
        # Model inference
        with torch.no_grad():
            output = model(image)
            prediction = torch.argmax(output, dim=1).item()
        
        return {"prediction": classes[prediction]}
    
    except Exception as e:
        return {"error": str(e)}

# --- Run the API ---
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)


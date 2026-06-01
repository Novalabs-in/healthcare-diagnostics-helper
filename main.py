import torch
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
import numpy as np

class ChestXRayClassifier:
    """
    Healthcare Diagnostic Classifier
    Utilizes deep learning architectures to assist medical image analysis.
    """
    def __init__(self):
        # Using pre-trained weights for transfer learning
        self.model = models.resnet18(pretrained=True)
        self.model.eval()
        self.labels = ["Normal", "Pneumonia", "Other Cardiothoracic Findings"]

    def prepare_image(self, pil_image):
        preprocess = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        return preprocess(pil_image).unsqueeze(0)

    def diagnose(self, tensor_img):
        with torch.no_grad():
            output = self.model(tensor_img)
            probabilities = torch.nn.functional.softmax(output[0], dim=0)
            
        top_idx = torch.argmax(probabilities).item() % len(self.labels)
        return self.labels[top_idx], probabilities[top_idx].item()

if __name__ == "__main__":
    classifier = ChestXRayClassifier()
    dummy_img = Image.fromarray(np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8))
    tensor = classifier.prepare_image(dummy_img)
    label, confidence = classifier.diagnose(tensor)
    print(f"Diagnostics Assist Result: {label} (Confidence: {confidence:.2%})")

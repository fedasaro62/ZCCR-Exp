import torch
import torchvision.models as models
from PIL import Image
from torchvision import transforms


class Classifier():
    def __init__(self, name='resnet50'):
        if name == 'resnet50':
            self.model  = models.resnet50(pretrained=True)

        self.model.eval()

        self.preprocess = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225])
        ])

    def classify(self, image: Image.Image):
        preprocessed     = self.preprocess(image)
        img_tensor       = torch.unsqueeze(preprocessed, 0)

        out              = self.model(img_tensor)
        with open('imagenet_classes.txt') as f:
            labels = [line.strip() for line in f.readlines()]

        _, index         = torch.max(out, 1)
        percentage       = torch.nn.functional.softmax(out, dim=1)[0] * 100

        # Print the top 5 scores along with the image label. Sort function is invoked on the torch to sort the scores.
        _, indices = torch.sort(out, descending=True)
        label = [(labels[idx], percentage[idx].item()) for idx in indices[0][:5]][0][0]
        class_, score = [(index[0].float(), percentage[idx].item()) for idx in indices[0][:5]][0]
        return class_, score, label
"""CelebModel architecture and preprocessing from the reference E5 notebook."""
import numpy as np
import torch
from torch import nn
from torchvision import transforms

# Official CelebA attribute order; an attribute-file header can verify/override it.
ATTRIBUTES = ['5_o_Clock_Shadow', 'Arched_Eyebrows', 'Attractive', 'Bags_Under_Eyes',
              'Bald', 'Bangs', 'Big_Lips', 'Big_Nose', 'Black_Hair', 'Blond_Hair',
              'Blurry', 'Brown_Hair', 'Bushy_Eyebrows', 'Chubby', 'Double_Chin',
              'Eyeglasses', 'Goatee', 'Gray_Hair', 'Heavy_Makeup', 'High_Cheekbones',
              'Male', 'Mouth_Slightly_Open', 'Mustache', 'Narrow_Eyes', 'No_Beard',
              'Oval_Face', 'Pale_Skin', 'Pointy_Nose', 'Receding_Hairline', 'Rosy_Cheeks',
              'Sideburns', 'Smiling', 'Straight_Hair', 'Wavy_Hair', 'Wearing_Earrings',
              'Wearing_Hat', 'Wearing_Lipstick', 'Wearing_Necklace', 'Wearing_Necktie', 'Young']
MEAN = [.5063, .4258, .3832]
STD = [.2644, .2436, .2397]


class Layer(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size=3, padding=1):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, kernel_size, 1, padding)
        self.bn = nn.BatchNorm2d(out_ch)
        self.relu = nn.ReLU()

    def forward(self, x):
        return self.relu(self.bn(self.conv(x)))


class CelebModel(nn.Module):
    def __init__(self):
        super().__init__()
        channels = [3, 32, 32, 64, 64, 128, 128, 128, 256, 256, 256, 512, 512, 512]
        for index in range(1, 14):
            kernel, padding = (5, 0) if 8 <= index <= 10 else (3, 0) if index >= 11 else (3, 1)
            setattr(self, f'unit{index}', Layer(channels[index-1], channels[index], kernel, padding))
        self.pool1, self.pool2, self.pool3, self.pool4 = [nn.MaxPool2d(2) for _ in range(4)]
        self.drop2, self.drop3 = nn.Dropout(.5), nn.Dropout(.5)
        self.pool5 = nn.AvgPool2d(2)
        self.model = nn.Sequential(self.unit1, self.unit2, self.pool1,
                                   self.unit3, self.unit4, self.pool2,
                                   self.unit5, self.unit6, self.unit7, self.pool3,
                                   self.unit8, self.unit9, self.unit10, self.pool4, self.drop2,
                                   self.unit11, self.unit12, self.unit13, self.pool5, self.drop3)
        self.fc = nn.Linear(512, 40)

    def forward(self, x):
        return self.fc(self.model(x).reshape(-1, 512))


def attribute_names(path):
    if path is None:
        return list(ATTRIBUTES)
    from pathlib import Path
    for line in Path(path).read_text().splitlines()[:3]:
        names = line.split()
        if names and names[0].lower() in {'image_id', 'filename'}:
            names = names[1:]
        if len(names) == 40 and set(names) == set(ATTRIBUTES):
            return names
    raise ValueError(f'Expected a CelebA 40-attribute header in {path}')


class CelebaAdapter:
    name = 'celeba_cnn'
    label_space = 'celeba_attributes'
    score_type = 'attribute logit'
    preprocessing_geometry = 'resize'

    def __init__(self, args, device):
        from model_adapters import read_class_names
        self.device, self.batch_size = device, args.model_batch_size
        self.class_names = read_class_names(args.class_names) or attribute_names(args.attribute_file)
        if len(self.class_names) != 40:
            raise ValueError('CelebModel requires exactly 40 attribute names')
        self.model = CelebModel()
        checkpoint = torch.load(args.checkpoint, map_location='cpu', weights_only=True)
        state = checkpoint.get('state_dict', checkpoint)
        state = {key.removeprefix('module.'): value for key, value in state.items()}
        self.model.load_state_dict(state, strict=True)
        self.model.to(device).eval()
        self.preprocess = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor(),
                                               transforms.Normalize(MEAN, STD)])

    def forward_tensor(self, x):
        return self.model(x)  # The notebook's XAI score is the raw attribute logit.

    def predict_batch(self, images):
        from PIL import Image
        rows = []
        with torch.inference_mode():
            for start in range(0, len(images), self.batch_size):
                batch = torch.stack([self.preprocess(Image.fromarray(np.asarray(image, dtype=np.uint8)))
                                     for image in images[start:start+self.batch_size]]).to(self.device)
                rows.append(self.model(batch).cpu().numpy())
        return np.concatenate(rows) if rows else np.empty((0, 40))

    def detections(self, image):
        return []

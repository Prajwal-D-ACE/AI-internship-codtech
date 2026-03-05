import torch
import torchvision.transforms as transforms
from torchvision.models import vgg19
from PIL import Image
import matplotlib.pyplot as plt

# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Image size
image_size = 256

# Image loader
loader = transforms.Compose([
    transforms.Resize((image_size, image_size)),
    transforms.ToTensor()
])

def load_image(path):
    image = Image.open(path).convert("RGB")
    image = loader(image).unsqueeze(0)
    return image.to(device)

content = load_image("content.jpg")
style = load_image("style.jpg")

# Load pretrained VGG19
vgg = vgg19(pretrained=True).features.to(device).eval()

# Initialize target image
target = content.clone().requires_grad_(True)

optimizer = torch.optim.Adam([target], lr=0.003)

content_weight = 1e5
style_weight = 1e10

# Helper function for gram matrix
def gram_matrix(tensor):
    b, c, h, w = tensor.size()
    tensor = tensor.view(c, h * w)
    gram = torch.mm(tensor, tensor.t())
    return gram

for step in range(300):

    target_features = vgg(target)
    content_features = vgg(content)
    style_features = vgg(style)

    content_loss = torch.mean((target_features - content_features) ** 2)

    gram_target = gram_matrix(target_features)
    gram_style = gram_matrix(style_features)

    style_loss = torch.mean((gram_target - gram_style) ** 2)

    total_loss = content_weight * content_loss + style_weight * style_loss

    optimizer.zero_grad()
    total_loss.backward()
    optimizer.step()

    if step % 50 == 0:
        print(f"Step {step} | Loss {total_loss.item()}")

# Save output
output = target.squeeze().cpu().detach()
plt.imsave("output.jpg", output.permute(1,2,0).numpy())

print("Stylized image saved as output.jpg")
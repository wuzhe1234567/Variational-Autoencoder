import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torch.amp import GradScaler, autocast
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import os


# Dice coefficient function - measures overlap between predicted and ground truth masks
def dice_coefficient(pred, target, num_classes, smooth=1e-6):
    dsc_scores = []
    for i in range(num_classes):
        pred_i = (pred == i).float()  # Convert predictions to binary for class i
        target_i = (target == i).float()  # Convert targets to binary for class i
        intersection = (pred_i * target_i).sum()  # Calculate intersection
        dsc = (2. * intersection + smooth) / (pred_i.sum() + target_i.sum() + smooth)  # Calculate Dice score
        dsc_scores.append(dsc.item())
    return dsc_scores


# Dice Loss function - custom loss based on Dice coefficient
class DiceLoss(nn.Module):
    def __init__(self, smooth=1e-6):
        super(DiceLoss, self).__init__()
        self.smooth = smooth

    def forward(self, inputs, targets):
        inputs = torch.sigmoid(inputs)  # Apply sigmoid to get probabilities
        inputs = torch.argmax(inputs, dim=1)  # Convert to class predictions
        inputs = inputs.view(-1)  # Flatten the tensor
        targets = targets.view(-1)  # Flatten the tensor

        intersection = (inputs == targets).sum()  # Calculate intersection
        union = inputs.size(0)  # Calculate union
        dice = (2. * intersection + self.smooth) / (union + self.smooth)  # Dice coefficient
        return 1 - dice  # Dice loss


# Combined Loss function - combination of Dice Loss and Cross-Entropy Loss
class CombinedLoss(nn.Module):
    def __init__(self, alpha=0.5):
        super(CombinedLoss, self).__init__()
        self.alpha = alpha  # Weighting factor for Dice Loss and Cross-Entropy Loss
        self.dice_loss = DiceLoss()  # Initialize Dice Loss
        self.cross_entropy_loss = nn.CrossEntropyLoss()  # Initialize Cross-Entropy Loss

    def forward(self, inputs, targets):
        targets = targets.squeeze(1)  # Remove channel dimension for compatibility
        # Combine Dice Loss and Cross-Entropy Loss
        return self.alpha * self.dice_loss(inputs, targets) + (1 - self.alpha) * self.cross_entropy_loss(inputs, targets)


# Dataset class for MR Images and corresponding masks
class MRImageDataset(Dataset):
    def __init__(self, image_dir, mask_dir, transform=None):
        self.image_dir = image_dir  # Directory for images
        self.mask_dir = mask_dir  # Directory for masks
        self.transform = transform  # Transformation for data augmentation
        # List all image and mask files
        self.image_files = sorted([f for f in os.listdir(image_dir) if f.endswith('.png')])
        self.mask_files = sorted([f for f in os.listdir(mask_dir) if f.endswith('.png')])

    def __len__(self):
        return len(self.image_files)  # Return the number of samples

    def __getitem__(self, idx):
        # Load image and mask files
        image_path = os.path.join(self.image_dir, self.image_files[idx])
        mask_path = os.path.join(self.mask_dir, self.mask_files[idx])

        image = Image.open(image_path).convert('L')  # Convert image to grayscale
        mask = Image.open(mask_path).convert('L')  # Convert mask to grayscale

        if self.transform:  # Apply transformation if available
            image = self.transform(image)
            mask = self.transform(mask)

        return image, mask  # Return image and corresponding mask


# Data transformations: Resize and convert to tensor
transform = transforms.Compose([
    transforms.Resize((256, 256)),  # Resize to 256x256
    transforms.ToTensor(),  # Convert to tensor
])

# Paths to training, validation, and test datasets
train_image_dir = r"C:\Users\wuzhe\Desktop\Study\COM3710\Lab2\keras_png_slices_data\keras_png_slices_data\keras_png_slices_seg_train"
train_mask_dir = r"C:\Users\wuzhe\Desktop\Study\COM3710\Lab2\keras_png_slices_data\keras_png_slices_data\keras_png_slices_train"
val_image_dir = r"C:\Users\wuzhe\Desktop\Study\COM3710\Lab2\keras_png_slices_data\keras_png_slices_data\keras_png_slices_seg_validate"
val_mask_dir = r"C:\Users\wuzhe\Desktop\Study\COM3710\Lab2\keras_png_slices_data\keras_png_slices_data\keras_png_slices_validate"
test_image_dir = r"C:\Users\wuzhe\Desktop\Study\COM3710\Lab2\keras_png_slices_data\keras_png_slices_data\keras_png_slices_seg_test"
test_mask_dir = r"C:\Users\wuzhe\Desktop\Study\COM3710\Lab2\keras_png_slices_data\keras_png_slices_data\keras_png_slices_test"

# Datasets and DataLoaders for training, validation, and testing
train_dataset = MRImageDataset(train_image_dir, train_mask_dir, transform=transform)
val_dataset = MRImageDataset(val_image_dir, val_mask_dir, transform=transform)
test_dataset = MRImageDataset(test_image_dir, test_mask_dir, transform=transform)

train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)  # Training DataLoader
val_loader = DataLoader(val_dataset, batch_size=4, shuffle=False)  # Validation DataLoader
test_loader = DataLoader(test_dataset, batch_size=4, shuffle=False)  # Testing DataLoader


# UNet model with Batch Normalization
class UNet(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(UNet, self).__init__()
        # Encoder: series of convolutional layers with BatchNorm and ReLU
        self.encoder1 = self.conv_block(in_channels, 64)
        self.encoder2 = self.conv_block(64, 128)
        self.encoder3 = self.conv_block(128, 256)
        self.encoder4 = self.conv_block(256, 512)
        self.encoder5 = self.conv_block(512, 1024)  # Additional encoder block

        self.pool = nn.MaxPool2d(2)  # Max pooling for downsampling
        # Decoder: transpose convolutions for upsampling and convolutional layers
        self.upconv5 = self.upconv(1024, 512)
        self.upconv4 = self.upconv(512, 256)
        self.upconv3 = self.upconv(256, 128)
        self.upconv2 = self.upconv(128, 64)

        # Decoding blocks: convolutional layers with concatenation from the encoder (skip connections)
        self.decoder5 = self.conv_block(1024, 512)
        self.decoder4 = self.conv_block(512, 256)
        self.decoder3 = self.conv_block(256, 128)
        self.decoder2 = self.conv_block(128, 64)

        # Final 1x1 convolution to produce the output with the required number of channels
        self.final = nn.Conv2d(64, out_channels, kernel_size=1)

    # Function to define a convolutional block with BatchNorm and ReLU activation
    def conv_block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    # Function for transpose convolution (upsampling)
    def upconv(self, in_channels, out_channels):
        return nn.Sequential(
            nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2),
            nn.ReLU(inplace=True)
        )

    # Forward pass through the network
    def forward(self, x):
        e1 = self.encoder1(x)
        e2 = self.encoder2(self.pool(e1))
        e3 = self.encoder3(self.pool(e2))
        e4 = self.encoder4(self.pool(e3))
        e5 = self.encoder5(self.pool(e4))

        d5 = self.upconv5(e5)
        d5 = torch.cat([d5, e4], dim=1)  # Skip connection from encoder to decoder
        d5 = self.decoder5(d5)

        d4 = self.upconv4(d5)
        d4 = torch.cat([d4, e3], dim=1)  # Skip connection
        d4 = self.decoder4(d4)

        d3 = self.upconv3(d4)
        d3 = torch.cat([d3, e2], dim=1)  # Skip connection
        d3 = self.decoder3(d3)

        d2 = self.upconv2(d3)
        d2 = torch.cat([d2, e1], dim=1)  # Skip connection
        d2 = self.decoder2(d2)

        return self.final(d2)  # Final output with required channels


# Function to display results: original image, ground truth mask, and predicted mask
def visualize_results(model, data_loader, device, num_images=4):
    model.eval()  # Set model to evaluation mode
    with torch.no_grad():  # Disable gradient computation
        images, masks = next(iter(data_loader))  # Get a batch of images and masks
        images = images.to(device)  # Move images to device (CPU/GPU)
        masks = masks.to(device)  # Move masks to device (CPU/GPU)
        outputs = model(images)  # Perform inference
        outputs = torch.argmax(outputs, dim=1)  # Convert model output to class predictions

        # Plot the original images, ground truth masks, and predicted masks
        for i in range(num_images):
            plt.figure(figsize=(10, 3))
            plt.subplot(1, 3, 1)
            plt.title("Image")
            plt.imshow(images[i].cpu().squeeze(), cmap='gray')
            plt.subplot(1, 3, 2)
            plt.title("Ground Truth")
            plt.imshow(masks[i].cpu().squeeze(), cmap='gray')
            plt.subplot(1, 3, 3)
            plt.title("Prediction")
            plt.imshow(outputs[i].cpu().squeeze(), cmap='gray')
            plt.show()


# Main function to train and validate the model
def train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs, device):
    scaler = GradScaler()  # Initialize mixed precision gradient scaler

    for epoch in range(num_epochs):
        model.train()  # Set model to training mode
        running_loss = 0.0
        for images, masks in train_loader:
            images, masks = images.to(device), masks.to(device)  # Move data to device

            with autocast(device_type='cuda', dtype=torch.float16):  # Use mixed precision
                outputs = model(images)  # Forward pass
                loss = criterion(outputs, masks)  # Compute loss

            scaler.scale(loss).backward()  # Backward pass with scaled loss
            scaler.step(optimizer)  # Perform optimization step
            scaler.update()  # Update the scaler
            optimizer.zero_grad()  # Reset gradients for next iteration

            running_loss += loss.item() * images.size(0)  # Accumulate loss

        # Calculate average loss over the epoch
        epoch_loss = running_loss / len(train_loader.dataset)

        # Validation phase
        model.eval()
        val_dice_scores = []
        with torch.no_grad():
            for images, masks in val_loader:
                images, masks = images.to(device), masks.to(device)
                outputs = model(images)
                outputs = torch.argmax(outputs, dim=1)  # Convert model output to class predictions
                dice_scores = dice_coefficient(outputs, masks, num_classes=4)  # Calculate Dice scores
                val_dice_scores.extend(dice_scores)

        val_dice_avg = np.mean(val_dice_scores)  # Calculate average validation Dice score
        print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {epoch_loss:.4f}, Validation Dice Score: {val_dice_avg:.4f}")

        # Optional: Implement early stopping based on validation dice score


if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # Use GPU if available
    model = UNet(in_channels=1, out_channels=4).to(device)  # Initialize model and move to device

    # Initialize optimizer and loss function
    optimizer = optim.AdamW(model.parameters(), lr=1e-4)
    criterion = CombinedLoss(alpha=0.5)  # Use combined Dice and Cross-Entropy loss

    # Train the model for 50 epochs
    num_epochs = 50
    train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs, device)

    # Visualize results on test data
    visualize_results(model, test_loader, device, num_images=4)

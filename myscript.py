import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, models, transforms
from torch.utils.data import DataLoader
import time
import os

def train_model(data_dir, num_epochs=25, batch_size=32, learning_rate=0.001, model_name='resnet18', use_pretrained=True):
    """
    Trains a convolutional neural network (CNN) for soil type classification using PyTorch.

    Args:
        data_dir (str): Path to the directory containing the dataset (e.g., 'dataset').
        num_epochs (int): Number of training epochs.
        batch_size (int): Batch size for training.
        learning_rate (float): Learning rate for the optimizer.
        model_name (str): The name of the pre-trained model to use ('resnet18', 'efficientnet_b0', etc.).
        use_pretrained (bool): Whether to use pre-trained weights.

    Returns:
        torch.nn.Module: The trained model.
    """
    # 1. Load and transform the data
    data_transforms = {
        'train': transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'test': transforms.Compose([  # Corrected: 'test' instead of 'val'
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
    }
    # Corrected:  Using 'test' set as is.  No separate validation.
    image_datasets = {'train': datasets.ImageFolder(os.path.join(data_dir, 'train'), data_transforms['train']),
                      'test': datasets.ImageFolder(os.path.join(data_dir, 'test'),  data_transforms['test'])}

    dataloaders = {'train': DataLoader(image_datasets['train'], batch_size=batch_size, shuffle=True, num_workers=4),
                   'test': DataLoader(image_datasets['test'], batch_size=batch_size, shuffle=False, num_workers=4)} # No shuffle for test
    dataset_sizes = {'train': len(image_datasets['train']), 'test': len(image_datasets['test']) }
    class_names = image_datasets['train'].classes # Get class names from the training data.

    # 2. Load a pre-trained model (ResNet18 or EfficientNet)
    if model_name == 'resnet18':
        model_ft = models.resnet18(pretrained=use_pretrained)
        num_ftrs = model_ft.fc.in_features
        model_ft.fc = nn.Linear(num_ftrs, len(class_names))  # Replace the last layer
    elif model_name == 'efficientnet_b0':
        model_ft = models.efficientnet_b0(pretrained=use_pretrained)
        num_ftrs = model_ft.classifier[1].in_features
        model_ft.classifier[1] = nn.Linear(num_ftrs, len(class_names))
    else:
        print(f"Model {model_name} not supported.  Using resnet18")
        model_ft = models.resnet18(pretrained=use_pretrained)
        num_ftrs = model_ft.fc.in_features
        model_ft.fc = nn.Linear(num_ftrs, len(class_names))

    # 3. Define loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer_ft = optim.Adam(model_ft.parameters(), lr=learning_rate)

    # 4. Train the model
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model_ft = model_ft.to(device)
    since = time.time()

    for epoch in range(num_epochs):
        print(f'Epoch {epoch}/{num_epochs - 1}')
        print('-' * 10)

        # Each epoch has a training and test phase  (Corrected: train and test)
        for phase in ['train', 'test']: # Corrected: train and test
            if phase == 'train':
                model_ft.train()  # Set model to training mode
            else:
                model_ft.eval()   # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0

            # Iterate over data
            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)

                # Zero the parameter gradients
                optimizer_ft.zero_grad()

                # Forward pass
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model_ft(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    # Backward pass + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer_ft.step()

                # Statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]

            print(f'{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')

        print()

    time_elapsed = time.time() - since
    print(f'Training complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')

    # 5. Evaluate on the test set
    model_ft.eval()  # Set to evaluation mode.  Important.
    test_loss = 0.0
    test_corrects = 0

    with torch.no_grad():
        for inputs, labels in dataloaders['test']:
            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = model_ft(inputs)
            _, preds = torch.max(outputs, 1)
            loss = criterion(outputs, labels)

            test_loss += loss.item() * inputs.size(0)
            test_corrects += torch.sum(preds == labels.data)

    test_loss = test_loss / dataset_sizes['test']
    test_acc = test_corrects.double() / dataset_sizes['test']
    print(f'Test Loss: {test_loss:.4f} Acc: {test_acc:.4f}')
    return model_ft

if __name__ == '__main__':
    data_dir = './dataset'  # Replace with the path to your dataset.
    num_epochs = 25
    batch_size = 32
    learning_rate = 0.001
    model_name = 'resnet18'  # or 'efficientnet_b0',  try both!
    use_pretrained = True # Use pre-trained weights

    # Check if data directory exists
    if not os.path.exists(data_dir):
        print(f"Error: Data directory '{data_dir}' not found.  Please create it and organize your data as shown below:")
        print("  dataset/")
        print("  ├── train/")
        print("  │   ├── alluvial soil/")
        print("  │   │   ├── image1.jpg")
        print("  │   │   ├── image2.jpg")
        print("  │   │   └── ...")
        print("  │   ├── black soil/")
        print("  │   │   ├── image3.jpg")
        print("  │   │   ├── image4.jpg")
        print("  │   │   └── ...")
        print("  │   ├── clay soil/")
        print("  │   │   ├── ...")
        print("  │   └── red soil/")
        print("  │       ├── ...")
        print("  └── test/")
        print("      ├── alluvial soil/")
        print("      │   ├── ...")
        print("      └── ... (black soil, clay soil, red soil)")
        exit()

    try:
        trained_model = train_model(data_dir, num_epochs, batch_size, learning_rate, model_name, use_pretrained)
        torch.save(trained_model.state_dict(), f'soil_classification_model_{model_name}.pth')
        print(f"Trained model ({model_name}) saved to soil_classification_model_{model_name}.pth")
    
    except Exception as e:
        print(f"An error occurred during training: {e}")
        print("Please check your data and code.")

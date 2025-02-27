import os
import torch
from torch.utils.data import DataLoader, Dataset, Subset, WeightedRandomSampler
from torchvision import datasets, transforms
import numpy as np
from sklearn.model_selection import train_test_split
from .utils import print_label_distribution

# Get the number of CPU cores available
num_workers = os.cpu_count()

# Define the class names 
class_names = ['basophil', 'eosinophil', 'erythroblast', 'ig', 'lymphocyte', 'monocyte', 'neutrophil', 'platelet']


# Transformation pipeline
transform = {
    'train': transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=15),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'val': transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'test': transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
}

# Define FilteredDataset class
class FilteredDataset(Dataset):
    def __init__(self, dataset, class_names):
        self.dataset = dataset
        self.class_to_idx = {cls: idx for idx, cls in enumerate(class_names)}
        self.filtered_indices = []
        self.targets = []
        
        print("Dataset classes before filtering:", dataset.classes)  

        # Filter and remap labels
        for idx, (_, label) in enumerate(dataset.samples):
            class_name = dataset.classes[label]
            if class_name in class_names:
                self.filtered_indices.append(idx)
                self.targets.append(self.class_to_idx[class_name])

        print(f"Number of samples after filtering: {len(self.filtered_indices)}")  

    def __len__(self):
        return len(self.filtered_indices)
    
    def __getitem__(self, idx):
        original_idx = self.filtered_indices[idx]
        image, _ = self.dataset[original_idx]
        label = self.targets[idx]
        return image, label

class TransformSubset(Dataset):
    def __init__(self, subset, transform=None):
        self.subset = subset
        self.transform = transform

    def __len__(self):
        return len(self.subset)

    def __getitem__(self, idx):
        image, label = self.subset[idx]
        if self.transform:
            image = self.transform(image)
        return image, label

# Function to create dataset and dataloaders
def create_dataloaders(dataset_dir, batch_size=16, num_workers=4):
    if not os.path.exists(dataset_dir):
        raise FileNotFoundError(f"Dataset directory '{dataset_dir}' not found.")

    # Load dataset
    full_dataset = datasets.ImageFolder(dataset_dir)

    # Filter dataset
    filtered_dataset = FilteredDataset(full_dataset, class_names)

    # Perform stratified split
    indices = np.arange(len(filtered_dataset))
    targets = np.array(filtered_dataset.targets)

    # First split: train and temp (val + test)
    train_idx, temp_idx = train_test_split(
        indices, 
        test_size=0.2,  # 40% for val+test
        stratify=targets,
        random_state=42
    )

    # Second split: val and test from temp
    val_idx, test_idx = train_test_split(
        temp_idx,
        test_size=0.5,  # Split temp into equal val and test
        stratify=targets[temp_idx],
        random_state=42
    )

    # ===== Add Class Weighted Sampler for Training =====
    # Compute sample weights for the training set based on class frequencies.
    train_targets = np.array(filtered_dataset.targets)[train_idx]
    unique_classes, class_counts = np.unique(train_targets, return_counts=True)

    # Inverse frequency for each class
    class_weights = {cls: 1.0 / count for cls, count in zip(unique_classes, class_counts)}

    # Create a weight for each sample in the training set
    sample_weights = np.array([class_weights[t] for t in train_targets])

    # Convert the sample_weights array to a torch tensor and then to a list
    sample_weights = torch.DoubleTensor(sample_weights).tolist()

    # Create the sampler; set replacement=True so that the sampling can pick
    # minority classes more often if necessary.
    train_sampler = WeightedRandomSampler(
        weights=sample_weights, 
        num_samples=len(sample_weights), 
        replacement=True
    )
    # ===================================================

    # Create subsets
    train_dataset = TransformSubset(Subset(filtered_dataset, train_idx), transform=transform['train'])
    val_dataset = TransformSubset(Subset(filtered_dataset, val_idx), transform=transform['val'])
    test_dataset = TransformSubset(Subset(filtered_dataset, test_idx), transform=transform['test'])

    # Create DataLoaders
    # When using a sampler (e.g., WeightedRandomSampler), do not specify shuffle=True.
# The sampler determines the order of samples, so setting shuffle is redundant and causes a conflict.
    train_loader = DataLoader(train_dataset, batch_size=batch_size, sampler=train_sampler, num_workers=num_workers)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, num_workers=num_workers)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, num_workers=num_workers)

    return train_loader, val_loader, test_loader, train_dataset, val_dataset, test_dataset

# Main block
if __name__ == "__main__":
    dataset_dir = '../data/PBC_dataset_normal_DIB_224/PBC_dataset_normal_DIB_224'
    train_loader, val_loader, test_loader, train_dataset, val_dataset, test_dataset = create_dataloaders(dataset_dir)

    # Verification
    print("\nDataset Splits:")
    print(f"Total dataset size: {len(train_dataset) + len(val_dataset) + len(test_dataset)}")
    print(f"Train size: {len(train_dataset)}")
    print(f"Val size: {len(val_dataset)}")
    print(f"Test size: {len(test_dataset)}")

    # Check label distribution
    print_label_distribution(train_loader, class_names, "Train")
    print_label_distribution(val_loader, class_names, "Validation")
    print_label_distribution(test_loader, class_names, "Test")

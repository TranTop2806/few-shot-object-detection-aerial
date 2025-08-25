import torch
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torch.utils.data import Dataset, DataLoader
import os
from PIL import Image
import torchvision.transforms as T
import json

def get_transform(train):
    """Data augmentation and transformations."""
    transforms = []
    transforms.append(T.ToTensor())
    if train:
        transforms.append(T.RandomHorizontalFlip(0.5))
    return T.Compose(transforms)

class NWPUDatasetFromJSON(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform

        self.base_class_names = [
            'ship', 'storage-tank', 'basketball-court',
            'ground-track-field', 'harbor', 'bridge', 'vehicle'
        ]
        self.class_name_to_model_idx = {name: idx + 1 for idx, name in enumerate(self.base_class_names)}
        print(f"Training base classes: {self.base_class_names}")

        self.image_infos = []
        self.annotations = {}

        for subset in ['train', 'test']:
            subset_dir = os.path.join(self.root_dir, subset)
            json_path = os.path.join(subset_dir, f'{subset}.json')

            if not os.path.exists(json_path):
                print(f"Skipping {subset}: annotation file not found")
                continue

            print(f"Loading {subset} annotations...")
            with open(json_path, 'r') as f:
                data = json.load(f)

            cat_id_to_name = {cat['id']: cat['name'] for cat in data.get('categories', [])}
            
            annotations_by_image = {}
            for ann in data.get('annotations', []):
                img_id = ann['image_id']
                if img_id not in annotations_by_image:
                    annotations_by_image[img_id] = []
                annotations_by_image[img_id].append(ann)

            for img_info in data.get('images', []):
                img_id = img_info['id']
                img_path = os.path.join(subset_dir, img_info['file_name'])

                if not os.path.exists(img_path):
                    continue
                
                img_annotations = annotations_by_image.get(img_id, [])
                base_class_anns = []
                for ann in img_annotations:
                    cat_name = cat_id_to_name.get(ann['category_id'])
                    if cat_name in self.class_name_to_model_idx:
                        bbox = ann['bbox']
                        x, y, w, h = bbox
                        box_coords = [x, y, x + w, y + h]
                        label_idx = self.class_name_to_model_idx[cat_name]
                        base_class_anns.append({'box': box_coords, 'label': label_idx})
                
                if base_class_anns:
                    self.image_infos.append({'path': img_path, 'id': img_id})
                    self.annotations[img_id] = base_class_anns
        print(f"Dataset loaded: {len(self.image_infos)} images")

    def __len__(self):
        return len(self.image_infos)
    
    def __getitem__(self, idx):
        img_info = self.image_infos[idx]
        img_path = img_info['path']
        img_id = img_info['id']
        
        image = Image.open(img_path).convert('RGB')
        annotations = self.annotations[img_id]
        
        boxes = [ann['box'] for ann in annotations]
        labels = [ann['label'] for ann in annotations]

        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        labels = torch.as_tensor(labels, dtype=torch.int64)

        target = {
            'boxes': boxes,
            'labels': labels,
            'image_id': torch.tensor([img_id])
        }
        
        if self.transform is not None:
            image = self.transform(image)
        
        return image, target

def collate_fn(batch):
    return tuple(zip(*batch))

def train_base_model():
    dataset_root = "data/NWPU VHR-10"
    
    if not os.path.isdir(dataset_root):
        print(f"Dataset root directory not found: {dataset_root}")
        return
        
    dataset = NWPUDatasetFromJSON(root_dir=dataset_root, transform=get_transform(train=True))
    
    if len(dataset) == 0:
        print("Failed to load any data. Aborting training.")
        return
    
    dataloader = DataLoader(
        dataset, 
        batch_size=4,
        shuffle=True, 
        collate_fn=collate_fn,
        num_workers=2
    )
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load a pre-trained Faster R-CNN model
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(weights=torchvision.models.detection.FasterRCNN_ResNet50_FPN_Weights.DEFAULT)
    
    num_classes = len(dataset.base_class_names) + 1  # N base classes + 1 background
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    
    model.to(device)
    
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=0.005, momentum=0.9, weight_decay=0.0005)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)
    
    num_epochs = 15
    model.train()
    
    print(f"Starting training for {num_epochs} epochs...")
    for epoch in range(num_epochs):
        epoch_loss = 0
        for i, (images, targets) in enumerate(dataloader):
            images = list(image.to(device) for image in images)
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
            
            loss_dict = model(images, targets)
            losses = sum(loss for loss in loss_dict.values())
            
            optimizer.zero_grad()
            losses.backward()
            optimizer.step()
            
            epoch_loss += losses.item()
            if i % 50 == 0:
                print(f"Epoch {epoch+1}/{num_epochs}, Batch {i}/{len(dataloader)}, Loss: {losses.item():.4f}")
        
        lr_scheduler.step()
        avg_epoch_loss = epoch_loss / len(dataloader)
        print(f"Epoch {epoch+1} completed. Average Loss: {avg_epoch_loss:.4f}. LR: {optimizer.param_groups[0]['lr']:.6f}")
    
    # Save the trained model
    output_dir = 'saved_models'
    os.makedirs(output_dir, exist_ok=True)
    torch.save(model.state_dict(), os.path.join(output_dir, 'base_model.pth'))
    print(f"Base model saved to {os.path.join(output_dir, 'base_model.pth')}")

if __name__ == '__main__':
    train_base_model()
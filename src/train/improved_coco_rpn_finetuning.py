import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torchvision.ops import box_iou
from PIL import Image
import numpy as np
import os
import logging
import json
import random
from sklearn.cluster import KMeans
from tqdm import tqdm

# DINOv2 Transform
dino_transform = torchvision.transforms.Compose([
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
])

# Augmentation transforms for few-shot data
augmentation_transform = torchvision.transforms.Compose([
    torchvision.transforms.RandomHorizontalFlip(p=0.5),
    torchvision.transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
])

class ImprovedPrototypeFinetuner:
    """
    Improved version with COCO pre-trained RPN and better fine-tuning strategies
    """
    def __init__(self, device, k_shot=20, num_bg_prototypes=30):
        self.device = device
        self.k_shot = k_shot
        self.num_bg_prototypes = num_bg_prototypes
        
        self.novel_classes = ['airplane', 'baseball diamond', 'tennis court']
        self.novel_class_to_idx = {name: idx for idx, name in enumerate(self.novel_classes)}
        
        self.base_model = self._load_base_model()
        self.dino_model = self._load_dino_model()
        print("Models loaded successfully.")

        self.object_prototypes = None
        self.background_prototypes = None
        self.all_prototypes = None

    def _load_base_model(self):
        """
        CRITICAL IMPROVEMENT: Use COCO pre-trained model for superior RPN
        Instead of our small NWPU-trained model
        """
        print("Loading Faster R-CNN model pre-trained on COCO for superior RPN...")
        # Use COCO pre-trained model - much better RPN than our small NWPU trained model
        model = torchvision.models.detection.fasterrcnn_resnet50_fpn(
            weights=torchvision.models.detection.FasterRCNN_ResNet50_FPN_Weights.DEFAULT
        )
        model.to(self.device)
        model.eval()  # Use only for proposal generation, not classification
        for param in model.parameters():
            param.requires_grad = False
        print("COCO pre-trained model loaded")
        return model

    def _load_dino_model(self):
        try:
            dino_model = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitl14', trust_repo=True)
            dino_model.to(self.device)
            dino_model.eval()
            for param in dino_model.parameters():
                param.requires_grad = False
            return dino_model
        except Exception as e:
            print(f"Failed to load DINOv2 model: {e}")
            raise

    @torch.no_grad()
    def _get_feature_map(self, image_pil):
        w, h = image_pil.size
        patch_size = self.dino_model.patch_size
        
        new_w, new_h = (w // patch_size) * patch_size, (h // patch_size) * patch_size
        if new_w == 0 or new_h == 0:
            print(f"Image size {w}x{h} is too small. Skipping.")
            return None, None
            
        resized_image = image_pil.resize((new_w, new_h))
        image_tensor = dino_transform(resized_image).unsqueeze(0).to(self.device)
        
        features_dict = self.dino_model.forward_features(image_tensor)
        patch_tokens = features_dict['x_norm_patchtokens']
        
        h_feat, w_feat = new_h // patch_size, new_w // patch_size
        
        expected_elements = h_feat * w_feat
        if patch_tokens.shape[1] != expected_elements:
             print(f"Shape mismatch! Expected {expected_elements} patch tokens, but got {patch_tokens.shape[1]}. Skipping image.")
             return None, None

        feature_map = patch_tokens.reshape(h_feat, w_feat, self.dino_model.embed_dim)
        feature_map = feature_map.permute(2, 0, 1).unsqueeze(0)
        return feature_map, (new_w, new_h)

    def _pool_features_from_map(self, feature_map, proposals, original_image_shape, resized_image_shape):
        if proposals.numel() == 0:
            return torch.empty((0, self.dino_model.embed_dim), device=self.device)

        orig_w, orig_h = original_image_shape
        resized_w, resized_h = resized_image_shape
        
        scaled_proposals = proposals.clone()
        scaled_proposals[:, 0::2] *= (resized_w / orig_w)
        scaled_proposals[:, 1::2] *= (resized_h / orig_h)

        patch_size = self.dino_model.patch_size
        pooled_features = torchvision.ops.roi_align(
            feature_map,
            [scaled_proposals],
            output_size=(1, 1),
            spatial_scale=1.0 / patch_size
        )
        return pooled_features.squeeze(-1).squeeze(-1)

    def prepare_few_shot_data(self, root_dir):
        print(f"Preparing {self.k_shot}-shot data for novel classes: {self.novel_classes}")
        
        annotations_by_class = {name: [] for name in self.novel_classes}
        all_image_paths = set()

        for subset in ['train', 'test']:
            subset_dir = os.path.join(root_dir, subset)
            json_path = os.path.join(subset_dir, f'{subset}.json')

            if not os.path.exists(json_path):
                print(f"JSON not found for subset '{subset}', skipping.")
                continue

            with open(json_path, 'r') as f:
                data = json.load(f)

            cat_id_to_name = {cat['id']: cat['name'] for cat in data.get('categories', [])}
            img_id_to_info = {img['id']: {'path': os.path.join(subset_dir, img['file_name']), 'w': img.get('width', 0), 'h': img.get('height', 0)} for img in data.get('images', [])}
            
            for ann in data.get('annotations', []):
                cat_name = cat_id_to_name.get(ann['category_id'])
                if cat_name in self.novel_classes:
                    img_info = img_id_to_info.get(ann['image_id'])
                    if img_info and os.path.exists(img_info['path']):
                        x, y, w, h = ann['bbox']
                        annotations_by_class[cat_name].append({
                            'image_path': img_info['path'],
                            'bbox': [x, y, x + w, y + h],
                            'image_shape': (img_info['w'], img_info['h'])
                        })
            
            for img_info in img_id_to_info.values():
                if os.path.exists(img_info['path']):
                    all_image_paths.add(img_info['path'])
        
        few_shot_data = {}
        for class_name, all_anns in annotations_by_class.items():
            if len(all_anns) < self.k_shot:
                print(f"Class '{class_name}' has only {len(all_anns)} samples, less than k={self.k_shot}. Using all available.")
                few_shot_data[class_name] = all_anns
            else:
                few_shot_data[class_name] = random.sample(all_anns, self.k_shot)
            print(f"Selected {len(few_shot_data.get(class_name, []))} shots for class '{class_name}'")

        return few_shot_data, list(all_image_paths)

    def create_prototypes(self, few_shot_data, all_image_paths):
        """Create initial object and background prototypes with augmentation."""
        print("Creating initial object prototypes with augmentation...")
        obj_prototypes_list = [[] for _ in self.novel_classes]
        
        for class_name, annotations in few_shot_data.items():
            if not annotations: continue
            
            class_idx = self.novel_class_to_idx[class_name]
            for ann in tqdm(annotations, desc=f"Processing {class_name}"):
                # Original image
                image = Image.open(ann['image_path']).convert('RGB')
                
                # Create multiple augmented versions for more robust prototypes
                for aug_idx in range(3):  # 3 augmented versions per sample
                    if aug_idx == 0:
                        # Original image
                        current_image = image
                    else:
                        # Augmented version
                        current_image = augmentation_transform(image)
                    
                    feature_map, resized_shape = self._get_feature_map(current_image)
                    if feature_map is None: continue

                    gt_box = torch.tensor([ann['bbox']], dtype=torch.float32).to(self.device)
                    pooled_feature = self._pool_features_from_map(feature_map, gt_box, ann['image_shape'], resized_shape)
                    obj_prototypes_list[class_idx].append(pooled_feature)

        valid_feats_by_class = [torch.cat(feats, dim=0) for feats in obj_prototypes_list if feats]
        if not valid_feats_by_class:
            raise ValueError("No valid features could be extracted for any novel class. Aborting.")
        
        self.object_prototypes = torch.stack([feats.mean(dim=0) for feats in valid_feats_by_class])
        print(f"Created object prototypes with shape: {self.object_prototypes.shape}")
        
        print("Creating background prototypes...")
        bg_features = []
        sampled_images = random.sample(all_image_paths, min(len(all_image_paths), 150))
        for img_path in tqdm(sampled_images, desc="Sampling backgrounds"):
            try:
                image = Image.open(img_path).convert('RGB')
                feature_map, _ = self._get_feature_map(image)
                if feature_map is None: continue
                
                _, _, h_feat, w_feat = feature_map.shape
                # Sample 3 random patch features from the feature map (reduced from 5)
                rand_h = torch.randint(0, h_feat, (3,))
                rand_w = torch.randint(0, w_feat, (3,))
                bg_features.append(feature_map[0, :, rand_h, rand_w].T)
            except Exception as e:
                print(f"Could not process background image {img_path}: {e}")

        if not bg_features:
            raise ValueError("Could not extract any background features.")

        all_bg_features = torch.cat(bg_features, dim=0)
        kmeans = KMeans(n_clusters=self.num_bg_prototypes, random_state=0, n_init='auto').fit(all_bg_features.cpu().numpy())
        self.background_prototypes = torch.tensor(kmeans.cluster_centers_, dtype=torch.float32).to(self.device)

        self.all_prototypes = torch.cat([self.object_prototypes, self.background_prototypes], dim=0)
        print(f"Prototypes created. Objects: {self.object_prototypes.shape}, Backgrounds: {self.background_prototypes.shape}")

    def finetune_prototypes(self, few_shot_data, num_epochs=50):
        if self.all_prototypes is None:
            raise RuntimeError("Prototypes have not been created. Call create_prototypes first.")
        
        print("Starting improved prototype finetuning...")
        self.all_prototypes = nn.Parameter(self.all_prototypes)
        
        optimizer = torch.optim.AdamW([self.all_prototypes], lr=5e-4, weight_decay=1e-4)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)
        loss_fn = nn.CrossEntropyLoss()

        for epoch in range(num_epochs):
            total_loss = 0
            num_samples = 0
            
            all_shots = [(ann, name) for name, anns in few_shot_data.items() for ann in anns if anns]
            random.shuffle(all_shots)

            for annotation, class_name in tqdm(all_shots, desc=f"Epoch {epoch+1}/{num_epochs}", leave=False):
                class_idx = self.novel_class_to_idx[class_name]
                
                image = Image.open(annotation['image_path']).convert('RGB')
                if random.random() > 0.5:  # 50% chance to apply augmentation
                    image = augmentation_transform(image)
                
                gt_boxes = torch.tensor([annotation['bbox']], dtype=torch.float32).to(self.device)

                with torch.no_grad():
                    feature_map, resized_shape = self._get_feature_map(image)
                    if feature_map is None: continue

                    # Use COCO pre-trained RPN for much better proposals
                    image_tensor_for_rpn = torchvision.transforms.ToTensor()(image).unsqueeze(0).to(self.device)
                    features_rpn = self.base_model.backbone(image_tensor_for_rpn)
                    proposals, _ = self.base_model.rpn(torchvision.models.detection.image_list.ImageList(image_tensor_for_rpn, [image_tensor_for_rpn.shape[-2:]]), features_rpn)
                    proposals = proposals[0]
                
                ious = box_iou(proposals, gt_boxes)
                
                pos_indices = torch.where(ious.max(dim=1).values >= 0.5)[0]
                neg_indices = torch.where(ious.max(dim=1).values < 0.2)[0]
                
                if len(pos_indices) == 0 or len(neg_indices) == 0: continue
                
                pos_indices = pos_indices[torch.randperm(len(pos_indices))[:8]]
                neg_indices = neg_indices[torch.randperm(len(neg_indices))[:24]]  # More negatives
                
                selected_proposals = torch.cat([proposals[pos_indices], proposals[neg_indices]], dim=0)
                proposal_features = self._pool_features_from_map(feature_map, selected_proposals, annotation['image_shape'], resized_shape)
                
                pos_labels = torch.full((len(pos_indices),), class_idx, dtype=torch.long, device=self.device)
                
                neg_features = proposal_features[len(pos_indices):]
                if neg_features.numel() > 0:
                    sim_to_bg = F.cosine_similarity(neg_features.unsqueeze(1), self.all_prototypes[len(self.novel_classes):].unsqueeze(0), dim=-1)
                    neg_labels = sim_to_bg.argmax(dim=1) + len(self.novel_classes)
                    labels = torch.cat([pos_labels, neg_labels], dim=0)
                else:
                    labels = pos_labels

                similarities = F.cosine_similarity(proposal_features.unsqueeze(1), self.all_prototypes.unsqueeze(0), dim=-1)
                logits = similarities / 0.05
                
                loss = loss_fn(logits, labels)
                
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
                num_samples += 1
            
            scheduler.step()
            avg_loss = total_loss / max(num_samples, 1)
            
            if (epoch + 1) % 10 == 0:
                print(f"Epoch [{epoch+1}/{num_epochs}], Avg Loss: {avg_loss:.4f}, LR: {optimizer.param_groups[0]['lr']:.6f}")

        # Normalize prototypes
        with torch.no_grad():
            self.all_prototypes.data = F.normalize(self.all_prototypes.data, p=2, dim=1)
        
        self.save_prototypes()

    def save_prototypes(self, filename='saved_models/improved_prototypes_coco_rpn_50epochs.pt'):
        output_dir = os.path.dirname(filename)
        os.makedirs(output_dir, exist_ok=True)
        torch.save({
            'prototypes': self.all_prototypes.detach(),
            'class_names': self.novel_classes,
            'method': 'improved_coco_rpn_augmentation'
        }, filename)
        print(f"Improved prototypes saved to {filename}")

def run_improved_finetuning():
    dataset_root = "data/NWPU VHR-10"
    
    if not os.path.isdir(dataset_root):
        print(f"Dataset root directory not found: {dataset_root}")
        return

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    finetuner = ImprovedPrototypeFinetuner(device, k_shot=20, num_bg_prototypes=30)
    
    few_shot_data, all_image_paths = finetuner.prepare_few_shot_data(root_dir=dataset_root)
    
    if not any(few_shot_data.values()):
        print("Could not find any data for the specified novel classes.")
        return
        
    finetuner.create_prototypes(few_shot_data, all_image_paths)
    
    finetuner.finetune_prototypes(few_shot_data, num_epochs=100)
    
if __name__ == '__main__':
    run_improved_finetuning()

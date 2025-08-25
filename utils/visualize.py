import torch
import torch.nn.functional as F
import torchvision
from PIL import Image, ImageDraw, ImageFont
import os
import random
import json
from tqdm import tqdm

dino_transform = torchvision.transforms.Compose([
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
])

class CocoRPNHelper:
    def __init__(self, device):
        self.device = device
        self.dino_model = self._load_dino_model()
        self.base_model = self._load_coco_model()
    
    def _load_coco_model(self):
        print("Loading COCO pre-trained Faster R-CNN...")
        model = torchvision.models.detection.fasterrcnn_resnet50_fpn(
            weights=torchvision.models.detection.FasterRCNN_ResNet50_FPN_Weights.DEFAULT
        )
        model.to(self.device)
        model.eval()
        for param in model.parameters():
            param.requires_grad = False
        return model
    
    def _load_dino_model(self):
        print("Loading DINOv2...")
        dino_model = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitl14', trust_repo=True)
        dino_model.to(self.device)
        dino_model.eval()
        for param in dino_model.parameters():
            param.requires_grad = False
        return dino_model
    
    @torch.no_grad()
    def _get_feature_map(self, image_pil):
        w, h = image_pil.size
        patch_size = self.dino_model.patch_size
        new_w, new_h = (w // patch_size) * patch_size, (h // patch_size) * patch_size
        if new_w == 0 or new_h == 0: 
            return None, None
        
        resized_image = image_pil.resize((new_w, new_h))
        image_tensor = dino_transform(resized_image).unsqueeze(0).to(self.device)
        features_dict = self.dino_model.forward_features(image_tensor)
        patch_tokens = features_dict['x_norm_patchtokens']
        h_feat, w_feat = new_h // patch_size, new_w // patch_size
        
        if patch_tokens.shape[1] != h_feat * w_feat: 
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
            feature_map, [scaled_proposals], output_size=(1, 1), spatial_scale=1.0 / patch_size
        )
        return pooled_features.squeeze(-1).squeeze(-1)

class CocoRPNDetector:
    def __init__(self, device, prototype_path='saved_models/improved_prototypes_coco_rpn_50epochs.pt'):
        self.device = device
        
        print("Loading models and prototypes...")
        self.helper = CocoRPNHelper(device)
        self.base_model = self.helper.base_model
        self.dino_model = self.helper.dino_model
        
        if not os.path.exists(prototype_path):
            print(f"Prototypes not found at {prototype_path}")
            print("Please run improved COCO RPN finetuning first!")
            raise FileNotFoundError(f"Prototypes not found at {prototype_path}")
        
        proto_data = torch.load(prototype_path, map_location=self.device)
        self.prototypes = proto_data['prototypes']
        self.novel_classes = proto_data['class_names']
        self.num_object_classes = len(self.novel_classes)
        self.idx_to_class = {i: name for i, name in enumerate(self.novel_classes)}
        
        print(f"Ready! Novel classes: {self.novel_classes}")
        print(f"Prototype shape: {self.prototypes.shape}")
    
    @torch.no_grad()
    def detect(self, image_path, conf_threshold=0.2, nms_threshold=0.3):
        print(f"Processing: {os.path.basename(image_path)}")
        
        try:
            image = Image.open(image_path).convert('RGB')
        except Exception as e:
            print(f"Could not open image: {e}")
            return [], [], []
        
        feature_map, resized_shape = self.helper._get_feature_map(image)
        if feature_map is None:
            print("Could not generate feature map")
            return [], [], []
        
        # Get proposals from COCO pre-trained RPN
        image_tensor_for_rpn = torchvision.transforms.ToTensor()(image).unsqueeze(0).to(self.device)
        features_rpn = self.base_model.backbone(image_tensor_for_rpn)
        proposals_list, _ = self.base_model.rpn(
            torchvision.models.detection.image_list.ImageList(
                image_tensor_for_rpn, [image_tensor_for_rpn.shape[-2:]]
            ), features_rpn
        )
        proposals = proposals_list[0]
        
        print(f"Generated {len(proposals)} proposals from COCO RPN")
        
        if len(proposals) > 300:
            proposals = proposals[:300]
        
        proposal_features = self.helper._pool_features_from_map(
            feature_map, proposals, image.size, resized_shape
        )
        
        if proposal_features.numel() == 0:
            print("No valid features")
            return [], [], []
        
        similarities = F.cosine_similarity(
            proposal_features.unsqueeze(1), 
            self.prototypes.unsqueeze(0), 
            dim=-1
        )
        scores, best_class_indices = torch.max(similarities, dim=1)
        
        is_object_mask = best_class_indices < self.num_object_classes
        is_confident_mask = scores >= conf_threshold
        keep_indices = torch.where(is_object_mask & is_confident_mask)[0]
        
        if len(keep_indices) == 0:
            print("No objects detected")
            return [], [], []
        
        final_boxes = proposals[keep_indices]
        final_scores = scores[keep_indices]
        final_labels_indices = best_class_indices[keep_indices]
        
        nms_indices = torchvision.ops.nms(final_boxes, final_scores, nms_threshold)
        
        result_boxes = final_boxes[nms_indices].cpu().tolist()
        result_scores = final_scores[nms_indices].cpu().tolist()
        result_labels = [self.idx_to_class[i.item()] for i in final_labels_indices[nms_indices]]
        
        print(f"Found {len(result_boxes)} objects after NMS")
        return result_boxes, result_scores, result_labels
    
    def visualize_detections(self, image_path, boxes, labels, scores, output_path):
        image = Image.open(image_path).convert('RGB')
        draw = ImageDraw.Draw(image)
        colors = ['#FF0000', '#00FF00', '#0000FF', '#FFFF00', '#FF00FF', '#00FFFF']
        
        for box, label, score in zip(boxes, labels, scores):
            class_idx = self.novel_classes.index(label)
            color = colors[class_idx % len(colors)]
            
            draw.rectangle(box, outline=color, width=3)
            text = f"{label}: {score:.3f}"
            
            try:
                font = ImageFont.load_default()
            except:
                font = ImageFont.load_default()
            
            text_bbox = draw.textbbox((box[0], box[1]), text, font=font)
            text_w, text_h = text_bbox[2] - text_bbox[0], text_bbox[3] - text_bbox[1]
            text_bg_y = box[1] - text_h - 4
            if text_bg_y < 0: 
                text_bg_y = box[1] + 4
            
            draw.rectangle(
                [box[0], text_bg_y, box[0] + text_w + 8, text_bg_y + text_h + 4], 
                fill=color
            )
            draw.text((box[0] + 4, text_bg_y + 2), text, fill="white", font=font)
        
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        image.save(output_path, quality=95)
        print(f"Saved to: {output_path}")

def find_test_images_with_novel_objects(test_json_path, novel_classes, max_images=30):
    with open(test_json_path, 'r') as f:
        data = json.load(f)
    
    cat_id_to_name = {cat['id']: cat['name'] for cat in data['categories']}
    novel_class_ids = {k for k, v in cat_id_to_name.items() if v in novel_classes}
    
    images_with_novel_objects = set()
    for ann in data['annotations']:
        if ann['category_id'] in novel_class_ids:
            images_with_novel_objects.add(ann['image_id'])
    
    img_id_to_info = {
        img['id']: {
            'id': img['id'],
            'path': os.path.join(os.path.dirname(test_json_path), img['file_name'])
        } 
        for img in data['images']
    }
    
    valid_images = []
    for img_id in images_with_novel_objects:
        if img_id in img_id_to_info:
            img_info = img_id_to_info[img_id]
            if os.path.exists(img_info['path']):
                valid_images.append(img_info)
    
    return valid_images[:max_images]

def main():
    dataset_root = "data/NWPU VHR-10"
    output_dir = "results_coco_rpn_visualize"
    test_json_path = os.path.join(dataset_root, "test", "test.json")
    
    if not os.path.isdir(dataset_root):
        print(f"Dataset not found: {dataset_root}")
        return
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    try:
        detector = CocoRPNDetector(device)
    except FileNotFoundError as e:
        print(e)
        return
    
    # Find test images
    test_images = find_test_images_with_novel_objects(test_json_path, detector.novel_classes, max_images=25)
    
    if not test_images:
        print("No test images found")
        return
    
    print(f"Visualizing {len(test_images)} images...")
    
    # Process images
    for i, img_info in enumerate(tqdm(test_images, desc="Processing images")):
        boxes, scores, labels = detector.detect(img_info['path'], conf_threshold=0.2)
        
        if boxes:
            output_filename = f"coco_rpn_result_{i+1:02d}_{os.path.basename(img_info['path'])}"
            output_path = os.path.join(output_dir, output_filename)
            detector.visualize_detections(img_info['path'], boxes, labels, scores, output_path)
    
if __name__ == '__main__':
    main()

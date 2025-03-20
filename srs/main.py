from multiprocessing import freeze_support
import os
import numpy as np
import torch
import albumentations as A
from torch.utils.data import DataLoader
from pycocotools.coco import COCO
from segmentation_models_pytorch import Unet
from segmentation_models_pytorch.losses import DiceLoss
from tqdm import tqdm
import matplotlib.pyplot as plt

if __name__ == '__main__':
    freeze_support()



# 1. Настройки
DATASET_DIR = "dataset"
ANNOTATIONS_PATH = os.path.join(DATASET_DIR, "result.json")
IMAGES_DIR = os.path.join(DATASET_DIR, "images")
BATCH_SIZE = 8
EPOCHS = 10
LEARNING_RATE = 1e-4
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# 2. Кастомный датасет
class COCOSegmentationDataset(torch.utils.data.Dataset):
    def __init__(self, root_dir, annotation_file, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.coco = COCO(annotation_file)
        self.image_ids = self.coco.getImgIds()
        
        # Фильтруем категории (только Road и Snow)
        self.cat_ids = self.coco.getCatIds(catNms=['Road', 'Snow'])
        self.load_category_mapping()

        # Фильтрация изображений с пустыми масками
        self.filtered_image_ids = []
        for img_id in self.image_ids:
            try:
                img_info = self.coco.loadImgs(img_id)[0]
                ann_ids = self.coco.getAnnIds(imgIds=img_info['id'], catIds=self.cat_ids)
                anns = self.coco.loadAnns(ann_ids)
                
                mask = np.zeros((img_info['height'], img_info['width']), dtype=np.uint8)
                for ann in anns:
                    if ann['category_id'] not in self.category_map:
                        continue
                    class_id = self.category_map[ann['category_id']]
                    mask = np.maximum(self.coco.annToMask(ann) * class_id, mask)
                
                if mask.sum() > 0:
                    self.filtered_image_ids.append(img_id)
            except Exception as e:
                print(f"Ошибка при фильтрации изображения {img_id}: {str(e)}")
        
        print(f"Исходное количество изображений: {len(self.image_ids)}")
        print(f"Количество изображений после фильтрации: {len(self.filtered_image_ids)}")

    def load_category_mapping(self):
        self.category_map = {cat_id: idx for idx, cat_id in enumerate(self.cat_ids)}
        print("Отображение category_id -> class_id:")
        for cat_id, class_id in self.category_map.items():
            print(f"category_id={cat_id} -> class_id={class_id}")

    def __getitem__(self, idx):
        img_id = self.filtered_image_ids[idx]
        img_info = self.coco.loadImgs(img_id)[0]
        img_path = os.path.join(self.root_dir, img_info['file_name'])
        
        image = plt.imread(img_path)
        
        ann_ids = self.coco.getAnnIds(imgIds=img_info['id'], catIds=self.cat_ids)
        anns = self.coco.loadAnns(ann_ids)
        
        mask = np.zeros((img_info['height'], img_info['width']), dtype=np.uint8)
        for ann in anns:
            if ann['category_id'] not in self.category_map:
                continue
            class_id = self.category_map[ann['category_id']]
            mask = np.maximum(self.coco.annToMask(ann) * class_id, mask)
        
        # Проверяем уникальные значения в маске
        unique_values = np.unique(mask)
        print(f"Уникальные значения в маске: {unique_values}")
        if not all(0 <= val < len(self.cat_ids) for val in unique_values):
            raise ValueError("Маска содержит недопустимые значения.")
        
        if self.transform:
            transformed = self.transform(image=image, mask=mask)
            image = transformed['image']
            mask = transformed['mask']
        
        image = torch.from_numpy(image).permute(2, 0, 1).float() / 255.0
        mask = torch.from_numpy(mask).long()
        
        return image, mask

    def __len__(self):
        return len(self.filtered_image_ids)
    
def load_category_mapping(self):
        self.category_map = {cat_id: idx for idx, cat_id in enumerate(self.cat_ids)}

# Функция для фильтрации батчей
def custom_collate_fn(batch):
    batch = [item for item in batch if item is not None]
    if len(batch) == 0:
        return None
    images, masks = zip(*batch)
    images = torch.stack(images)
    masks = torch.stack(masks)
    return images, masks

def __getitem__(self, idx):
    try:
        img_info = self.coco.loadImgs(self.image_ids[idx])[0]
        img_path = os.path.join(self.root_dir, img_info['file_name'])
        
        # Проверяем существование файла
        if not os.path.exists(img_path):
            raise FileNotFoundError(f"Файл не найден: {img_path}")
            
        image = plt.imread(img_path)
        
        # Создаем маску
        ann_ids = self.coco.getAnnIds(imgIds=img_info['id'], catIds=self.cat_ids)
        anns = self.coco.loadAnns(ann_ids)
        
        mask = np.zeros((img_info['height'], img_info['width']), dtype=np.uint8)
        for ann in anns:
            if ann['category_id'] not in self.category_map:
                continue
            class_id = self.category_map[ann['category_id']]
            mask = np.maximum(self.coco.annToMask(ann) * class_id, mask)
        
        # Если маска пустая
        if mask.sum() == 0:
            mask = np.zeros_like(mask)
        
        # Применяем аугментации
        if self.transform:
            transformed = self.transform(image=image, mask=mask)
            image = transformed['image']
            mask = transformed['mask']
        
        # Преобразуем в тензоры
        image = torch.from_numpy(image).permute(2, 0, 1).float() / 255.0
        mask = torch.from_numpy(mask).long()
        
        return image, mask
    
    except Exception as e:
        print(f"Ошибка при загрузке изображения {idx}: {str(e)}")
        return None

# 3. Аугментации
# Аугментации
transform = A.Compose([
    A.Resize(512, 512),  # Приводим все изображения к размеру 512x512
    A.HorizontalFlip(p=0.5),
    A.RandomBrightnessContrast(p=0.2),
    A.Rotate(limit=10, p=0.3),
    A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
])


# 4. Загрузка данных
train_dataset = COCOSegmentationDataset(
    root_dir=IMAGES_DIR,
    annotation_file=ANNOTATIONS_PATH,
    transform=transform
)

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=0,
    collate_fn=custom_collate_fn
)

# 5. Модель
model = Unet(
    encoder_name="resnet34",
    encoder_weights="imagenet",
    classes=2,  # Только Road и Snow
    activation=None
).to(DEVICE)

# 6. Функция потерь и оптимизатор
criterion = DiceLoss(mode='multiclass')
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

# 7. Цикл обучения
def train_epoch(model, loader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    total_batches = 0  # Счетчик обработанных батчей

    for batch in tqdm(loader, desc="Training"):
        if batch is None:  # Пропускаем пустые батчи
            continue
        
        images, masks = batch
        images = images.to(device)
        masks = masks.to(device)
        
        # Обнуляем градиенты
        optimizer.zero_grad()
        
        # Прямой проход
        outputs = model(images)
        
        # Вычисляем loss
        loss = criterion(outputs, masks)
        running_loss += loss.item()
        
        # Обратный проход
        loss.backward()
        optimizer.step()

        total_batches += 1  # Увеличиваем счетчик батчей
    
    # Если не было обработано ни одного батча, возвращаем NaN
    return running_loss / total_batches if total_batches > 0 else float('nan')

# 8. Валидация
def validate(model, loader, criterion, device):
    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for images, masks in tqdm(loader, desc="Validation"):
            images = images.to(device)
            masks = masks.to(device)
            
            outputs = model(images)
            loss = criterion(outputs, masks)
            val_loss += loss.item()
    
    return val_loss / len(loader)

# 9. Основной цикл
for epoch in range(EPOCHS):
    print(f"Epoch {epoch+1}/{EPOCHS}")
    train_loss = train_epoch(model, train_loader, criterion, optimizer, DEVICE)
    print(f"Train Loss: {train_loss:.4f}")
    torch.save(model.state_dict(), f"model_epoch_{epoch+1}.pth")
    
    # Сохранение чекпоинта
    torch.save(model.state_dict(), f"model_epoch_{epoch+1}.pth")

# 10. Визуализация результатов
def visualize_prediction(model, dataloader, device, num_samples=3):
    model.eval()
    with torch.no_grad():
        for i, (images, masks) in enumerate(dataloader):
            if i >= num_samples:
                break
            images = images.to(device)
            outputs = model(images)
            preds = torch.argmax(outputs, dim=1).cpu().numpy()
            
            # Отображаем оригинал, маску и предсказание
            plt.figure(figsize=(15, 5))
            plt.subplot(1, 3, 1)
            plt.title("Image")
            plt.imshow(images[0].permute(1, 2, 0).cpu().numpy())
            
            plt.subplot(1, 3, 2)
            plt.title("True Mask")
            plt.imshow(masks[0].numpy())
            
            plt.subplot(1, 3, 3)
            plt.title("Predicted Mask")
            plt.imshow(preds[0])
            plt.show()

# Визуализируем предсказания на тренировочных данных
visualize_prediction(model, train_loader, DEVICE)
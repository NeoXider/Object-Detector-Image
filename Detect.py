from transformers import DetrImageProcessor, DetrForObjectDetection
import torch
from PIL import Image, ImageDraw, ImageFont
import requests
import os

def load_image_from_url(url):
    """Загружает изображение по URL."""
    return Image.open(requests.get(url, stream=True).raw)

def load_image_from_path(path):
    """Загружает изображение по пути."""
    return Image.open(path)

def resize_image(image, scale_factor):
    """Уменьшает изображение на заданный коэффициент."""
    new_size = (image.width // scale_factor, image.height // scale_factor)
    return image.resize(new_size)

def load_model(model_path=None):
    """Загружает предобученную модель и процессор."""
    processor = DetrImageProcessor.from_pretrained("facebook/detr-resnet-50")
    model = DetrForObjectDetection.from_pretrained("facebook/detr-resnet-50")
    
    if model_path and os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path))
    model.half()
    
    return processor, model

def detect_objects(image, processor, model, device):
    """Обнаруживает объекты на изображении и возвращает результаты."""
    inputs = processor(images=image, return_tensors="pt").to(device)
    inputs = {k: v.half() for k, v in inputs.items()}  # Convert inputs to FP16
    outputs = model(**inputs)
    target_sizes = torch.tensor([image.size[::-1]]).to(device).half()  # Convert target_sizes to FP16
    results = processor.post_process_object_detection(outputs, target_sizes=target_sizes, threshold=0.9)[0]
    return results

def draw_boxes(image, results, model):
    """Рисует прямоугольники вокруг обнаруженных объектов на изображении."""
    draw = ImageDraw.Draw(image)
    for score, label, box in zip(results["scores"], results["labels"], results["boxes"]):
        box = [round(i, 2) for i in box.tolist()]
        draw.rectangle(box, outline="red", width=2)  # Рисуем прямоугольник
        font_size = 24
        text_x = round(box[0]) + 5
        text_y = round(box[3])
        draw.text((text_x, text_y), f"{model.config.id2label[label.item()]}: {round(score.item(), 2)}", fill="red", font=ImageFont.truetype("arial.ttf", font_size))  # Добавляем текст
    return image
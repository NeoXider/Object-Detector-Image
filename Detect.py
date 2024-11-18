from transformers import DetrImageProcessor, DetrForObjectDetection
import torch
from PIL import Image, ImageDraw, ImageFont
import requests
import os

def load_image_from_url(url):
    """Load an image from a URL."""
    response = requests.get(url, stream=True)
    response.raise_for_status()
    return Image.open(response.raw)

def load_image_from_path(path):
    """Load an image from a file path."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"The file {path} does not exist.")
    return Image.open(path)

def resize_image(image, scale_factor):
    """Resize the image by a given scale factor."""
    new_size = (image.width // scale_factor, image.height // scale_factor)
    return image.resize(new_size)

def load_model(model_path=None):
    """
    Load a pre-trained model and processor.
    
    :param model_path: Path to a local model checkpoint (optional).
    :return: Tuple containing the image processor and model.
    """
    processor = DetrImageProcessor.from_pretrained("facebook/detr-resnet-50")
    model = DetrForObjectDetection.from_pretrained("facebook/detr-resnet-50")
    
    if model_path and os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path))
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device).half()
    
    return processor, model

def detect_objects(image, processor, model, device):
    """
    Detect objects in the given image.
    
    :param image: PIL Image object.
    :param processor: DETR image processor.
    :param model: DETR model.
    :param device: Device (e.g., 'cuda' or 'cpu') to run the model on.
    :return: Detection results.
    """
    inputs = processor(images=image, return_tensors="pt").to(device)
    inputs = {k: v.half() for k, v in inputs.items()}  # Convert inputs to FP16
    outputs = model(**inputs)
    target_sizes = torch.tensor([image.size[::-1]]).to(device).half()  # Convert target_sizes to FP16
    results = processor.post_process_object_detection(outputs, target_sizes=target_sizes, threshold=0.9)[0]
    return results

def draw_boxes(image, results, model):
    """
    Draw bounding boxes around detected objects on the image.
    
    :param image: PIL Image object.
    :param results: Detection results from DETR.
    :param model: DETR model used for detection.
    :return: Image with drawn bounding boxes and labels.
    """
    draw = ImageDraw.Draw(image)
    font_size = 24
    font = ImageFont.truetype("arial.ttf", font_size) if os.path.exists("arial.ttf") else None
    
    for score, label, box in zip(results["scores"], results["labels"], results["boxes"]):
        box = [round(i, 2) for i in box.tolist()]
        draw.rectangle(box, outline="red", width=2)  # Draw the bounding box
        text_x = round(box[0]) + 5
        text_y = round(box[1]) - font_size if font else round(box[3])
        label_name = model.config.id2label[label.item()]
        draw.text((text_x, text_y), f"{label_name}: {round(score.item(), 2)}", fill="red", font=font)  # Add text
        
    return image
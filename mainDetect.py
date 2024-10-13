from flask import Flask, request, jsonify, render_template, redirect, url_for
from Detect import *
import torch
from PIL import Image
import io
import threading

app = Flask(__name__)

# Загрузка модели при запуске сервера
processor, model = load_model("detr_resnet50_fp16.pth")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

print(f"Model is running on device: {device}")

# Переменная для масштабирования изображения
scale_factor = 2

# Глобальная переменная для хранения результатов
results_store = {}

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/about')
def about():
    return render_template('about.html')

@app.route('/loading')
def loading():
    task_id = request.args.get('task_id')
    return render_template('loading.html', task_id=task_id)

@app.route('/detect', methods=['POST'])
def detect():
    client_ip = request.remote_addr
    print("Start Detect from IP: " + client_ip)

    if 'image' not in request.files:
        print("No image provided")
        return jsonify({"error": "No image provided"}), 400

    image_file = request.files['image']
    show_image = 'show_image' in request.form
    task_id = str(len(results_store) + 1)

    print("Starting image processing...")
    
    # Запускаем обработку изображения в отдельном потоке
    thread = threading.Thread(target=process_image_task, args=(image_file, show_image, task_id))
    thread.start()

    # Перенаправляем на страницу ожидания с task_id
    print("Redirecting to loading page...")
    return redirect(url_for('loading', task_id=task_id))

def process_image_task(image_file, show_image, task_id):
    image_bytes = image_file.read()
    image = Image.open(io.BytesIO(image_bytes))
    if image.format == 'WEBP':
        image = image.convert("RGB")
    if image.mode != "RGB":
        image = image.convert("RGB")

    image = resize_image(image, scale_factor)

    print("Processing image...")
    try:
        results = detect_objects(image, processor, model, device)
    except Exception as e:
        results_store[task_id] = {"error": f"Detection failed: {str(e)}"}
        return

    detections = []
    for score, label, box in zip(results["scores"], results["labels"], results["boxes"]):
        box = [round(i, 2) for i in box.tolist()]
        detections.append({
            "label": model.config.id2label[label.item()],
            "confidence": round(score.item(), 3),
            "box": box
        })

    if show_image:
        image_with_boxes = draw_boxes(image, results, model)
        image_with_boxes.save(f"static/output_with_boxes_{task_id}.jpg")
        results_store[task_id] = {"detections": detections, "image_url": f"static/output_with_boxes_{task_id}.jpg"}
    else:
        results_store[task_id] = {"detections": detections}

    print("Processing complete for task:", task_id)

@app.route('/task_status/<task_id>')
def task_status(task_id):
    result = results_store.get(task_id)
    if result:
        return jsonify(result)
    else:
        return jsonify({"status": "processing"}), 202

@app.route('/detect_url', methods=['POST'])
def detect_url():
    client_ip = request.remote_addr
    print("Start detect_url from IP: " + client_ip)

    url = request.form.get('url')
    show_image = 'show_image' in request.form
    if not url:
        return jsonify({"error": "No URL provided"}), 400

    task_id = str(len(results_store) + 1)
    thread = threading.Thread(target=process_url_task, args=(url, show_image, task_id))
    thread.start()

    # Перенаправляем на страницу ожидания с task_id
    print("Redirecting to loading page...")
    return redirect(url_for('loading', task_id=task_id))

def process_url_task(url, show_image, task_id):
    try:
        image = load_image_from_url(url)
        if image.format == 'WEBP':
            image = image.convert("RGB")
        if image.mode != "RGB":
            image = image.convert("RGB")
    except Exception as e:
        results_store[task_id] = {"error": f"load url: {str(e)}"}

    image = resize_image(image, scale_factor)

    print("Processing URL image...")
    try:
        results = detect_objects(image, processor, model, device)
    except Exception as e:
        results_store[task_id] = {"error": f"Detection failed: {str(e)}"}

    detections = []
    for score, label, box in zip(results["scores"], results["labels"], results["boxes"]):
        box = [round(i, 2) for i in box.tolist()]
        detections.append({
            "label": model.config.id2label[label.item()],
            "confidence": round(score.item(), 3),
            "box": box
        })

    if show_image:
        image_with_boxes = draw_boxes(image, results, model)
        image_with_boxes.save(f"static/output_with_boxes_{task_id}.jpg")
        results_store[task_id] = {"detections": detections, "image_url": f"static/output_with_boxes_{task_id}.jpg"}
    else:
        results_store[task_id] = {"detections": detections}

    print("Processing complete for task:", task_id)

@app.route('/results')
def results():
    task_id = request.args.get('task_id')
    result = results_store.get(task_id)
    if not result:
        return render_template('404.html'), 404
    if 'error' in result:
        return render_template('error.html', error=result['error'])
    return render_template('results.html', detections=result.get('detections', []))

@app.route('/results_with_image')
def results_with_image():
    task_id = request.args.get('task_id')
    result = results_store.get(task_id)
    if not result:
        return render_template('404.html'), 404
    if 'error' in result:
        return render_template('error.html', error=result['error'])
    return render_template('results_with_image.html', detections=result.get('detections', []), image_url=result.get('image_url', ''))

@app.errorhandler(404)
def page_not_found(e):
    return render_template('404.html'), 404

@app.route('/error')
def error():
    message = request.args.get('message', 'An error occurred')
    return render_template('error.html', error=message)

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000)
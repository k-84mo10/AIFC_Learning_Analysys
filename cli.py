import os
import configparser
import ast
import torch
import torchvision.datasets as datasets
from torch.utils.data import DataLoader
from PIL import Image, ImageDraw, ImageFont
import numpy as np

from AIFC_Learning_Analysys.lib.machine_learning import MachineLearning

def main():

    config_path = os.path.join(os.path.dirname(__file__), "config", "config.ini")
    config = configparser.ConfigParser()
    config.read(config_path)
    
    device = torch.device(
        config.getint("device", "gpu_number") if torch.cuda.is_available() else "cpu"
    )
    print(f"{device} is used as device.")

    model_path = config.get("model", "model_path")
    model_type = config.get("model", "model_type")

    machine_learning = MachineLearning(model_type)
    machine_learning.load_model(model_path)
    machine_learning.model = machine_learning.model.to(device)
    print(f"model is sent to {device}.")

    dataset_path = config.get("dataset", "dataset_path")
    image_dataset = datasets.ImageFolder(root=dataset_path, transform=machine_learning.transform)
    dataloader = DataLoader(image_dataset, batch_size=32, shuffle=True, num_workers=4)

    class_names = image_dataset.classes
    print(f'Loaded {len(image_dataset)} images in {len(class_names)} classes.')

    result_dir = "AIFC_Learning_Analysys/result"
    os.makedirs(result_dir, exist_ok=True)
    output_dir = "AIFC_Learning_Analysys/result/misclassified_images"
    os.makedirs(output_dir, exist_ok=True)

    output_file = "AIFC_Learning_Analysys/result/AIFC_Learning_Analysys_output.csv"
    with open(output_file, "w") as file:
        file.write("image_path, predict, predict_value, label, label_value, ")
        for class_value in class_names:
            file.write(f"'{class_value}, ")
        file.write("\n")

    # Perform inference on all batches
    all_predictions = []
    all_outputs = []
    all_labels = []
    for batch_idx, (images, labels) in enumerate(dataloader):
        images = images.to(device)
        outputs = machine_learning.inference(images)
        all_outputs.extend(outputs.cpu().numpy())
        predicted_classes = torch.argmax(outputs, dim=1)
        all_predictions.extend(predicted_classes.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

        # Check for misclassified images
        for i in range(len(images)):
            if predicted_classes[i] != labels[i]:
                # Save the misclassified image
                image_np = images[i].cpu().numpy().transpose((1, 2, 0))
                image_np = (image_np * 0.5 + 0.5) * 255.0  # De-normalize the image
                image_np = image_np.astype(np.uint8)
                image_pil = Image.fromarray(image_np)

                # Get the original image size
                width, height = image_pil.size

                # Create a new image with white background to place the original image and text
                new_width = width + 100
                new_image = Image.new("RGB", (new_width, height), (255, 255, 255))
                new_image.paste(image_pil, (0, 0))

                # Draw the predicted and actual labels on the right side
                draw = ImageDraw.Draw(new_image)
                font = ImageFont.load_default()
                text = f"Pred: {class_names[predicted_classes[i]]} ({outputs[i][predicted_classes[i]]:.2f})\nLabel: {class_names[labels[i]]} ({outputs[i][labels[i]]:.2f})\n"
                text_y_position = 10

                for line in text.split('\n'):
                    draw.text((width + 10, text_y_position), line, font=font, fill=(0, 0, 0))
                    text_y_position += 20

                # Save the image
                output_path = os.path.join(output_dir, f"AIFC_Learning_Analysys_misclassified_{batch_idx}_{i}.png")
                new_image.save(output_path)

                with open(output_file, "a") as file:
                    file.write(f"AIFC_Learning_Analysys_misclassified_{batch_idx}_{i}.png, '{class_names[predicted_classes[i]]}, {outputs[i][predicted_classes[i]]:.2f}, '{class_names[labels[i]]}, {outputs[i][labels[i]]:.2f},")
                    for value in outputs[i]:
                        file.write(f"{value:.2f}, ")
                    file.write("\n")

                print(f"Saved misclassified image to {output_path}")
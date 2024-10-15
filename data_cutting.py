import os
from ultralytics import YOLO
from PIL import Image
import shutil

model = YOLO('C:/Users/shyn/Documents/Project/data_cutting_for_image_classification/best.pt')

def save_cropped_images(results, save_dir, image_name):
    crops = results[0].boxes.xyxy  
    
    image_pil = Image.open(image_name)

    for i, box in enumerate(crops):

        xmin, ymin, xmax, ymax = map(int, box)

 
        cropped_img = image_pil.crop((xmin, ymin, xmax, ymax))


        base_name = os.path.splitext(os.path.basename(image_name))[0]
        cropped_img.save(os.path.join(save_dir, f"{base_name}_crop_{i}.jpg"))


def process_directory(input_dir, output_dir):

    for root, dirs, files in os.walk(input_dir):

        relative_path = os.path.relpath(root, input_dir)
        new_dir = os.path.join(output_dir, relative_path + "_after_cut")
        os.makedirs(new_dir, exist_ok=True)

        for file in files:
            if file.endswith(('.jpg', '.jpeg', '.png', '.bmp', '.tiff')):  
                img_path = os.path.join(root, file)
                

                results = model(img_path)
                

                save_cropped_images(results, new_dir, img_path)


def main(input_folder):

    base_folder_name = os.path.basename(os.path.normpath(input_folder))
    output_folder = f"{base_folder_name}_after_cut"

    if os.path.exists(output_folder):
        shutil.rmtree(output_folder)

    process_directory(input_folder, output_folder)
    print(f"Completed. Cropped images are saved in {output_folder}")

input_folder = 'C:/Users/shyn/Documents/Project/data_cutting_for_image_classification/data1/Young_Healthy' 
main(input_folder)

# Apple Freshness Segmentation with YOLOv8

This project demonstrates semantic segmentation for apple freshness using the YOLOv8 model. The segmentation task classifies pixels into three classes: fresh apple, rotten apple, and rotten part.

## Dataset

- Dataset is hosted on Roboflow and contains polygon annotations for three classes:
  - **fresh-apple**
  - **rotten-apple**
  - **rotten-part**
- [Link to dataset](https://app.roboflow.com/fruitclassification-djarn/fruit_segmentation-jibtl/1/export)

## Project Structure

- `notebooks/`: Contains the Colab notebook for segmentation with YOLOv8.
- `data/`: Stores the dataset with annotations.
- `results/`: Directory for storing training results and predictions.
- `model/`: YOLOv8 model checkpoints after training.

## Requirements

- Python 3.x
- YOLOv8 (installed via the `ultralytics` package)
- Google Colab and Google Drive for file storage

## Installation

1. Clone the repository:
    ```bash
    git clone https://github.com/yourusername/your-repo-name.git
    cd your-repo-name
    ```

2. Run the following command in the Colab notebook to install YOLOv8:
    ```python
    !pip install ultralytics
    ```

## Usage

The following steps walk you through training and testing the model on Google Colab.

1. **Mount Google Drive**: This is necessary to access and save files in Colab.
   ```python
   from google.colab import drive
   drive.mount('/content/drive')
   ```

2. **Set Up and Train YOLOv8 Model**:
   - Define the model and configure the dataset location, training for 10 epochs with a 640-pixel image size.
   ```python
   import os
   from ultralytics import YOLO
   
   ROOT_DIR = '/content/drive/MyDrive/yolov8_003'
   model = YOLO("yolov8n-seg.yaml")
   results = model.train(data=os.path.join(ROOT_DIR, "data.yaml"), epochs=10, imgsz=640)
   ```

3. **Validate the Model**:
   - Use validation to retrieve metrics like mAP (Mean Average Precision) for both bounding boxes and segmentation masks.
   ```python
   metrics = model.val()
   print("Box mAP:", metrics.box.map)  # Mean mAP (50-95) for bounding boxes
   print("Segmentation mAP:", metrics.seg.map)  # Mean mAP (50-95) for masks
   ```

4. **Inference and Predictions**:
   - Run predictions on test images and display or save the results.
   ```python
   results = model.predict("/content/drive/MyDrive/yolov8_003/fresh1.jpg", save=True, save_txt=True, imgsz=640, conf=0.5)
   ```

5. **Calculate Rotten Part Percentage**:
   - Using the segmentation masks, calculate the percentage of the rotten area relative to the total area of each apple.
   ```python
   import numpy as np

   def calculate_area(mask):
       return np.sum(mask) if isinstance(mask, np.ndarray) else 0

   # Example with results from model prediction
   results = model.predict("/content/drive/MyDrive/yolov8_003/rotten01.jpg", save=True, imgsz=640, conf=0.3)
   
   masks = results[0].masks
   class_ids = results[0].boxes.cls
   rotten_apple_area = sum(calculate_area(mask) for i, mask in enumerate(masks) if int(class_ids[i]) == 1)
   rotten_part_area = sum(calculate_area(mask) for i, mask in enumerate(masks) if int(class_ids[i]) == 2)

   if rotten_apple_area > 0:
       rotten_percentage = (rotten_part_area / rotten_apple_area) * 100
       print(f"Rotten part percentage: {rotten_percentage:.2f}%")
   else:
       print("No rotten apple detected.")
   ```

6. **Save Results to Google Drive**:
   - Save the output and model files to your Google Drive for easy access and sharing.
   ```python
   import shutil, os
   destination = '/content/drive/MyDrive/yolov8_003_results'
   if os.path.exists(destination):
       shutil.rmtree(destination)
   shutil.copytree('/content/runs/', destination)
   ```

## Example Output

- **Classes**: fresh apple, rotten apple, rotten part
- **Metrics**:
  - Box mAP: 
  - Segmentation mAP:
  
Sample predictions:
```
Object type: fresh-apple
Coordinates: [50, 30, 200, 180]
Confidence: 0.90
--
Object type: rotten-part
Coordinates: [120, 60, 300, 220]
Confidence: 0.80
```

## Results

- The model can accurately segment fresh apples, rotten apples, and rotten parts.
- Example images of segmented outputs and metric scores will be saved in the `results/` directory.
## Training and validation of modle
![val_batch2_pred](https://github.com/user-attachments/assets/c5f958ce-ba88-4d99-baaf-f707c42a929a)
![val_batch2_labels](https://github.com/user-attachments/assets/4fd228cd-7e7f-44a4-b3b7-6c4f45fe2830)
![val_batch1_pred](https://github.com/user-attachments/assets/f0dd0674-a542-4984-af7e-84d5c4f6cb9d)
![val_batch1_labels](https://github.com/user-attachments/assets/4aa2ec20-462f-4831-a0d2-1f0f189d5324)
![val_batch0_pred](https://github.com/user-attachments/assets/bd21c595-55b4-438a-acc7-19669206a424)
![val_batch0_labels](https://github.com/user-attachments/assets/6ab4261b-b9a6-4e6c-b40c-d42edc99b88e)

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

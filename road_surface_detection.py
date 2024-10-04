# import shutil
# from google.colab import files

# shutil.make_archive('/content/Road-Surface-Classification-2', 'zip', '/content/datasets/Road-Surface-Classification-2')
# files.download('/content/Road-Surface-Classification-2.zip')

from ultralytics import YOLO
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

# Load a YOLO model
model = YOLO("yolov8n.pt")

# Train the model with the correct file path format for Windows
train_results = model.train(
    data=r"C:\Users\harik\Downloads\Project_BTP\Road-Surface-Classification\data.yaml", 
    epochs=25, 
    imgsz=640,  
    device="cpu",  
    save=True,  
    project=r"C:\Users\harik\Downloads\Project_BTP\Results"
)


# Evaluate the model performance on the validation set
metrics = model.val()

# Assuming metrics['pred'] and metrics['true'] contain the predictions and true labels
# Get the predictions and true labels from validation set results
y_pred = metrics.pred  # You may need to extract these based on how YOLOv8 returns results
y_true = metrics.true  # True labels from validation data

# Compute the confusion matrix


conf_matrix = confusion_matrix(y_true, y_pred)

# Display the confusion matrix
disp = ConfusionMatrixDisplay(confusion_matrix=conf_matrix)
disp.plot(cmap=plt.cm.Blues)

plt.title("Confusion Matrix")
plt.show()

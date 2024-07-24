# YOLOv10 on custom dataset for Obj. Detection
1. collect data and do labeling with [roboflow](https://www.google.com)
2. Train your custom dataset on [google colab](https://colab.research.google.com/github/roboflow-ai/notebooks/blob/main/notebooks/train-yolov10-object-detection-on-custom-dataset.ipynb#scrollTo=s5RGYA6sPgEd)
3. Execute it from start to finish
4. This section is optional, you can skip it and go straight to “download dataset from roboflow universe”.
   ![image](https://github.com/user-attachments/assets/f729cdd1-769d-42bb-9072-63b8b878f2a2)
5. Download the "best.pt" file and use it in "yolov10test.py", to use it, run this command on terminal:
   ```bsh
   pip install ultralytics
7. If you get this error: "AttributeError: Can't get attribute 'v10DetectLoss' on <module 'ultralytics.utils.loss'>", follow instruction below. Otherwise congratulations you have successfully created object detection with YOLOv10.

## v10DetectLoss Error
1. Try to upgrade your ultralytics using:
   ```bsh
   pip install --upgrade ultralytics
2. if the problem is still the same and unresolved, we use the awikwok method.
3. Go to this directory `C:\Users\"username"\AppData\Local\Programs\Python\Python312\Lib\site-packages\ultralytics\nn`
4. Edit `task.py` refer to this site (https://github.com/ultralytics/ultralytics/pull/13977/files)

# <u>Autonomous Reforestation Robot (ML Model) for PDE4433_CW2</u>

This project develops a machine learning model aimed at predicting the most suitable crops for desert environments, designed for integration with an Autonomous Reforestation Robot. The model processes sensor data, including soil moisture, humidity, nitrogen, potassium, and phosphorus levels, to determine the optimal crop type.

By utilizing sensor-collected data, the model provides accurate predictions, thereby enhancing the precision of crop recommendations for sustainable agriculture.

**Folder Structure**<br>
![Folder_Structure](src/images/folder_structure.png)

1. data/ : To store all the data that use for model training and testing
2. models/ : For save all the models.
3. notebooks/ : All Jupyter notebooks used for programming.
4. src/: All the source data (Ex. images, videos) will save here.

## Data sets
For the model training, I have used below dataset.
1. Image Datasets.
- https://www.kaggle.com/datasets/prasanshasatpathy/soil-types
- https://www.kaggle.com/datasets/jhislainematchouath/soil-types-dataset

<b>Sample view of final dataset of soil textures</b>
![DataSet](src/images/soilDataset.png)

2. The following datasets were taken from Kaggle.
- https://www.kaggle.com/datasets/varshitanalluri/crop-recommendation-dataset

<b>Sample view of tabular dataset</b>
![DataSet](src/images/dataSet.png)

## <u>Models</u>
The system employs a two-stage modeling approach where the first model analyzes soil texture images to classify soil type, while the second model integrates this output with additional sensor data to predict suitable crop types. This modular design ensures specialized processing for each data modality while maintaining interoperability between components.

In this stage ML model will analyse the sensor data including predicted data from first model, and then predict suitable crop type for the area. 
<a href="notebooks/PDE4433_CW2_FinalModelTraining.ipynb">(Model Analysis)</a> 

Models and accuracy so far: 
1. <a href="models/soilRecognizeModel/soil_clasify_model_epc15.h5">First Model trained by CNN</a>
    - Training Accuracy: 90.36%
    - Test Accuracy: 85.00%
2. <a href="models/Crop_RecommendationDT_decision_tree_model.pkl">Second Model trained by Decision Tree</a>
    - Train Accuracy: 95.39% 
    - Test Accuracy: 95.00%
2. <a href="models/Crop_RecommendationDT_random_forest_model.pkl">Second Model trained by Random Forest Model</a>
    - Train Accuracy: 99.35%
    - Test Accuracy: 99.24%

<br>

The soil prediction model was trained using Google Colab. Codes and test files can be see <a href="https://drive.google.com/drive/folders/1S1gEy1sYb-HPSGwDsGU8mo7gU-cSxGoU?usp=sharing">here</a>.


Soil recognition model:
![DataSet](src/images/soilTrainModel.png)

Soil texture prediction:
![DataSet](src/images/imagePredict.png)


Due to its superior accuracy, the Random Forest Model was selected for further robotics development.

## <u>Challenges</u>
### 1. Find datasets
- Due to the specialized nature of the conducted research and studies, identifying a suitable dataset proved to be a significant challenge. As the focus of this analysis is to examine desert areas with abundant tree coverage, finding a relevant dataset specific to this domain was particularly difficult. As a result, commonly used datasets for model training were considered as alternative options

### 2. Avoid overfitting and increase predict accuracy
- The dataset used for model training is a tabular dataset, with the target output being of a categorical data type. For model training, Decision Tree and Random Forest algorithms were employed. To mitigate the risk of overfitting and enhance prediction accuracy, training was halted at an optimal point, with extensive fine-tuning applied to achieve the best model performance.

## <u>Future enhancement</u>
1. Expanding the dataset with diverse soil textures and climate attributes will improve model accuracy, especially in desert regions.
2. Advanced deep learning techniques like Transformers and CNN-RNN hybrids will enhance feature extraction and adaptability.
3. Real-time learning through online machine learning will enable continuous model improvement with new data.
4. Direct integration of sensor outputs will support autonomous planting, irrigation, and soil monitoring robots.
5. pLiDAR, GPS, and sensor fusion will improve navigation and precision in desert terrains.
6. Satellite imagery, drones, and GIS will enhance soil analysis and large-scale reforestation planning.
7. IoT and cloud integration will enable real-time monitoring, remote control, and scalable data processing.
8. Solar-powered and energy-efficient robotics will ensure sustainable operation in remote areas.
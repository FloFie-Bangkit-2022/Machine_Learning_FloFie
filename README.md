# Machine_Learning_FloFie
Everything about Machine Learning Part in FloFie App.

## Dataset
We use dataset from tensorflow dataset named `tf_flower`. The dataset contains 3.670 files with 5 class. Class in the dataset are:
1. Dandelion
2. Daisy
3. Tulips
4. Sunflowers
5. Roses

More info about the dataset: https://www.tensorflow.org/datasets/catalog/tf_flowers

## Model
We use pre-trained and fine-tuning model. The pre-trained we use is `EfficientNetV2B1`. The reason is because in the keras application (https://keras.io/api/applications/) it has 95% accuracy in top-5 accuracy, small size model (34 Mb) and parameter less than 10M (8.9 M). Our model after fine-tuned has 98% accuracy and with the size model in tflite around 26 Mb. 

More info about `EfficientNetV2B1`: https://arxiv.org/abs/2104.00298.

## Deployment
We use convert model to tflitebecause we want to deploy our model in android app. You can find our tflite model in https://drive.google.com/drive/folders/1dHInVtIIrqn0rwiHRWluVcPaXYMGfTdN?usp=sharing or in branch -> master -> flower.tflite

## Additional 
Please read Tutorial.md if you want to know about our model and our model project name is Flowers_EfficientNetV2B1.ipynb. 

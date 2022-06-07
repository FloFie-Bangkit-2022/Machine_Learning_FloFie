# Machine_Learning_FloFie
Repository for machine learning team.

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
We use convert model to tflite cause its more easy to do.

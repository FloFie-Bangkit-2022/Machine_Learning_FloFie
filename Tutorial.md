# Model Documentations

Eveything about Flower-EfficientNetV2B1 Model

# Table of contents

1. [Introduction](#introduction)
2. [Library](#library)
3. [Prepare The Data](#data)
   1. [Dataset](#dataset)
   2. [Create Directory](#directory)
   3. [Resize Image](#resize)
   4. [Optional](#optional)
4. [Callback](#callback)
5. [Transfer Learning](#transfer_learning)
   1. [Import Pre-Trained Model](#pretrained_model)
   2. [Freeze Layer in Model](#freeze_layer)
   3. [Build Model](#build_model)
   4. [Compile Model](#compile_model)
   5. [Training Model Transfer Learning](#training_model_transfer)
   6. [Evaluate Model Transfer Learning](#evaluate_model_transfer)
6. [Fine-Tuning](#fine_tuning)
   1. [Compile and Training Fine-Tuned Model](#compile_training_fine_tuned)
7. [Predict and Testing](#predict_testing)
   1. [Test a Model from Capture Image](#test_using_camera)
8. [Classification Report](#classification_report)
9. [Deploy Model](#deploy_model)
	1. [Convert to TFLite](#convert_tflie)
10. [Additional (Code)](#additional_code)

## This is the introduction <a name="introduction"></a>

This document is about steps of Flower-EfficientNetV2B1 Model in FloFie app projects. FloFie app is Bangkit Capstone Project's. The Model is for classification flowers image.

## Library <a name="library"></a>

Library that used to model are

1. Tensorflow (https://www.tensorflow.org/)
2. Tensorflow Dataset (https://www.tensorflow.org/datasets/)
3. Keras (https://keras.io/)
4. Pandas (https://pandas.pydata.org/)
5. Numpy (https://numpy.org/)
6. Matplotlib (https://matplotlib.org/)
7. Sklearn (https://scikit-learn.org/stable/)
8. IPython (https://ipython.readthedocs.io/en/stable/)
9. Seaborn (https://seaborn.pydata.org/)
10. Os (https://docs.python.org/3/library/os.html)

## Prepare The Data <a name="data"></a>

Prepare your data first before training model.

### Dataset<a name="Dataset"></a>

Before our model training, make sure to import dataset in code. In this model, dataset that used is `tf_flower` from tensorflow dataset. To use dataset from tensorflow dataset you can use `tfds.load`. You can find in this article https://www.tensorflow.org/datasets/api_docs/python/tfds/load to understand how to use tfds.load. Make sure to include split (to split the dataset), as_supervised (to get labels name), and with_info (to get info about the dataset) in tfds.load.

### Create Directory<a name="directory"></a>

Create directory for your model that save later and for label.

### Resize Image<a name="resize"></a>

Image that were taken or uploaded usually have different length and width. So, you need to resize image to make it model easier to understand. Also, you need to resize image based on criteria from pre-trained model. You need to read carefully pre-trained model needed. Some model need to be spesific for input model. You can find criteria for preprossing image in `keras application` or in `Article Publication`. In this model we use EfficientNetV2B1 so we resize image to 240x240 and change image type to float32 for neural network.

### Optional<a name="optional"></a>

This part of prepare-data is optional. You can add it in your model or not.

1.  Input Pipeline (to help model run effficiently)
2.  Data Augmentation (make variant data from the existing data). It is optional but I suggest you to add this process if your data is little or to increase accuracy.

## Callback<a name="callback"></a>

A callback is an object that can perform actions at various stages of training (e.g. at the start or end of an epoch, before or after a single batch, etc) (https://keras.io/api/callbacks/). You can choose different and more than one variant callback for your training model. In this model, the callback are model checkpoint and early stopping.

## Transfer Learning<a name="transfer_learning"></a>

Preparation for to make transfer learning.

### Import Pre-Trained Model<a name="pretrained_model"></a>

To make transfer learning and fine tuning model you need to import pre-trained model that you want. You can find existing model in https://keras.io/api/applications/. Remember to read carefully the model that you choose because there is some information about model that can use for prepare the data. For EfficientNetV2B1 you can find to how to import this in https://keras.io/api/applications/efficientnet_v2/#efficientnetv2b1-function.

### Freeze Layer<a name="freeze_layer"></a>

You need to freeze the layer so you can avoid destroying any of the information they contain during future training rounds. You can choose which layer that you want to be frozen. In this EfficientNetV2B1 model we freeze the trainable layer.

### Build Model<a name="build_model"></a>

You still need to make your own model to match with your problem. Also dont forget to include the pre-trained model with layer that have been frozen. In this EfficientNetV2B1 model we make model with 7 layers include data augmentation, freeze layer, and others.

### Compile Model<a name="compile_model"></a>

Before training make sure to choose loss, optimizer, and metrics properly which match with your model. In this EfficientNetV2B1 model we use SparseCategoricalCrossentropy() because we are not doing one-hot encoding. Use Adam as optimizer and accuracy as metrics.

### Training Model Transfer Learning<a name="training_model_transfer"></a>

Now, It is time to do training your own model. Make sure to choose your epoch, steps_per_epoch and etc.

### Evaluate Model Transfer Learning<a name="evaluate_model_transfer"></a>

Evaluate your model with validation data or test data. you can use model.evaluate(data that you want testing)

## Fine-tuning Model<a name="fine_tuning"></a>

After the model that we made, we can improve the accuracy with fine-tuning. In the fine tuning make sure to make our trainable parameter in pre-trained model to True (`base_model.trainable = True`). and make sure to make BatchNormalization to keep false because it can reduce accuracy.

### Compile and Training Fine-Tuned Model<a name="compile_training_fine_tuned"></a>

Compile the model as same with before and then training again the model also evaluate the model. The code is similar with transfer learning model before.

## Predict and Testing<a name="predict_testing"></a>

You can try to predict the test data with model.predict(test_data) It is similar with evaluate but different. Model.evaluate() return accuracy. Model.predict() return the prediction based on what model learn.

### Test a Model from Capture Image<a name="test_using_camera"></a>

You can test your model in colab using camerato take capture of image and return the label. For testing using camera in colab you can find in `code snippet` and then find camera capture. Make sure to add some other code so that after image captured it can return label.

## Classification Report<a name="classification_report"></a>

You can use classification report to make sure that your model work properly and can be comparison with other model. You can find more about this in https://scikit-learn.org/stable/modules/generated/sklearn.metrics.classification_report.html.

## Deploy Model<a name="deploy_model"></a>

After you make your model you can deploy for production in app, web, microcontroller and others. You can model using TF-Serving or TF-Lite. You can find more about tf-serving in https://www.tensorflow.org/tfx/tutorials/serving/rest_simple. For tflite in https://www.tensorflow.org/tfx/tutorials/tfx/tfx_for_mobile. Additional link https://www.tensorflow.org/lite/inference_with_metadata/lite_support for understand how to add tflite in android app.

### Convert to TFLite<a name="convert_tflie"></a>

Choose tflite if you want model on-edge device. After covert model to tflite and save it at the directory that we make before. You can download it and put in `assest` folder in android app. You can find our tflite model at branch->Master->flower.tflite.

## Additional (Code)<a name="additional_code"></a>

If you want to learn the code you can open our Flowers EfficientNetV2B1.ipynb file or https://www.tensorflow.org/tutorials/images/transfer_learning.



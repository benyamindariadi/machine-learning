# Machine Learning

## Getting Started

First of all, We should say thank you to [Benyamin Dariadi](https://github.com/benyamindariadi). With a lot of his effort during Pandemic, He can create this repository and spread the knowledge about machine learning.

Also other contributors that tried to revamp his repository and add some of their works. So in the future we can take a look this repository for learning purposes.

## Prerequisites

What things you need to use it

```bash
1. Jupyter Notebook
2. Anaconda / pip
```

---

## Contents

- [Classification](#classification) - Projects related to datasets that have the purpose of predicting a state or class
- [Forecasting](#forecasting) - projects related to models that aim to predict a continuous value
- [Image](#image) - projects related to image datasets that are processed with deep learning
- [Text](#text) - With some topics about Natural language processing
- [Computer Vision](#computervision) - With topics about image and video processing (mostly use OpenCV)

---

## Classification

### 1. [Categorical Peoples Interests](https://github.com/benyamindariadi/machine-learning/tree/master/clustering-categorical-peoples-interests)

- **Context:** There are 4 groups/classes of people and 217 hobby's and interest questions
- **Project:** Predict the person's class based on their interests.
- **Algorithms:** Random Forest, Decision Tree, K-Nearest Neighbors, Artificial Neural Networks.
- **Dataset:** [Source.](https://www.kaggle.com/rainbowgirl/clustering-categorical-peoples-interests)

### 2. [Mushroom Classification](https://github.com/benyamindariadi/machine-learning/tree/master/mushroom-classification)

- **Context:** This dataset includes descriptions of hypothetical samples corresponding to 23 species of gilled mushrooms in the Agaricus and Lepiota Family Mushroom drawn from The Audubon Society Field Guide to North American Mushrooms (1981). Each species is identified as definitely edible, definitely poisonous, or of unknown edibility and not recommended. This latter class was combined with the poisonous one. The Guide clearly states that there is no simple rule for determining the edibility of a mushroom; no rule like "leaflets three, let it be'' for Poisonous Oak and Ivy.
- **Project:** Predict whether the mushroom is poisonous or not.
- **Algorithms:** Random Forest, Decision Tree, K-Nearest Neighbors, Support Vector Machine.
- **Dataset:** [Source.](https://www.kaggle.com/uciml/mushroom-classification)

### 3. [Fake Job Description Prediction](https://github.com/benyamindariadi/machine-learning/tree/master/real-or-fake-fake-jobposting-prediction)

- **Context:** The data consists of both textual information and meta-information about the jobs. The dataset can be used to create classification models which can learn the job descriptions which are fraudulent.
- **Project:** Predict whether the job posting is real or not
- **Algorithms:** Decision Tree, K-Nearest Neighbors, Support Vector Machine, Artificial Neural Networks.
- **Dataset:** [Source.](https://www.kaggle.com/shivamb/real-or-fake-fake-jobposting-prediction)

### 4. [Twitter User Gender Classification](https://github.com/benyamindariadi/machine-learning/tree/master/twitter-user-gender-classification)

- **Context:** This data set was used to train a CrowdFlower AI gender predictor. You can read all about the project here. Contributors were asked to simply view a Twitter profile and judge whether the user was a male, a female, or a brand (non-individual). The dataset contains 20,000 rows, each with a user name, a random tweet, account profile and image, location, and even link and sidebar color.
- **Project:** Predict twitter user's gender
- **Algorithms:** Decision Tree, K-Nearest Neighbors, Support Vector Machine.
- **Dataset:** [Source.](https://www.kaggle.com/crowdflower/twitter-user-gender-classification)
- **Note:** In this project I neglect the text column, please look at [Text](#text) section my model based on user's tweets and profile descriptions

### 5. [Online Shoppers Intention](https://github.com/benyamindariadi/machine-learning/tree/master/online-shoppers-intention)

- **Context:** The dataset consists of feature vectors belonging to 12,330 sessions. The dataset was formed so that each session would belong to a different user in a 1-year period to avoid any tendency to a specific campaign, special day, user profile, or period.
- **Project:** Predict whether the customer will generate revenue or not
- **Algorithms:** Random Forest, Support Vector Machine, Artificial Neural Networks.
- **Dataset:** [Source.](https://www.kaggle.com/roshansharma/online-shoppers-intention)

### 6. [Fake GPS Detection (GOJEK)](https://github.com/benyamindariadi/machine-learning/tree/master/Fake%20GPS%20Detection%20(GOJEK))

- **Context:** Among our drivers, there are drivers who use Fake GPS application to mock their location. This FGPS usage is unfair for other GOJEK drivers who work honestly. Hence, we would like to apply a machine learning model to classify whether a trip is being done using fake GPS or not based on their PING behavior.
- **Project:** Builds models to predict fake order. 
- **Algorithms:** Decision Tree,Random Forest, Support Vector Machine, K-Nearest Neighbors, Artificial Neural Networks.
- **Dataset:** [Source.](https://www.kaggle.com/c/dsbootcamp10/data)
- **Note:** The dataset consists of the order_id which is the unique key. Each order id consists of many data rows representing pings which will give 8 different informations (columns). So the 'usefull' informations (data) must be extracted from the pings on each order id. Data extracting dan preparation will located on df_train_cleaning.ipynb, and the cleaned datased saved on clean_df_train.csv. The ML's model performances will provide on MLs.ipynb.

- #### [UPDATE](https://github.com/benyamindariadi/machine-learning/tree/master/Fake%20GPS%20Detection%20(GOJEK))

- **Project:** Builds models to predict fake order with **PyCaret** library.(pycaret.ipynb)

### 7. [Marketing Project (Online Deployed)](https://github.com/benyamindariadi/machine-learning/tree/master/Project-Marketing)

- **Context:** The data is related with direct marketing campaigns of a Portuguese banking institution. The marketing campaigns were based on phone calls. Often, more than one contact to the same client was required, in order to access if the product (bank term deposit) would be ('yes') or not ('no') subscribed.
- **Project:** Predict whether the potential clients will be subscribe a term deposit or not.
- **Dataset:** [Source.](https://archive.ics.uci.edu/ml/datasets/Bank+Marketing)
- **DEPLOYMENT:** [**WEB APP**](https://bank-marketing-app.herokuapp.com/)

### 8. [Weather Dataset](https://github.com/benyamindariadi/machine-learning/tree/master/weather%20dataset)

- **Context:** This dataset contains daily weather observations from numerous Australian weather stations.
- **Project:** Predict whether tomorrow will be rain or not. I build this model with **PyCaret** library.
- **Dataset:** [Source.](https://www.kaggle.com/jsphyg/weather-dataset-rattle-package)

### 9. [Fake News Classification](https://github.com/benyamindariadi/machine-learning/tree/master/fake-news-classification)

- **Context:** This project uses a csv based fake news datasets
- **Project:** Classify whether the input is fake news or not.
- **Algorithm:** LSTM.
- **Dataset:** [Source.](https://www.datacamp.com/community/tutorials/scikit-learn-fake-news)

## Forecasting

### 1. [Hotel Booking Demand](https://github.com/benyamindariadi/machine-learning/tree/master/hotel-booking-demand)

- **Context:** This data set contains booking information for a city hotel and a resort hotel, and includes information such as when the booking was made, length of stay, the number of adults, children, and/or babies, and the number of available parking spaces, among other things.
- **Project:** Predict the number of booking.
- **Algorithm:** Recurrent Neural Networks (LSTM).
- **Dataset:** [Source.](https://www.kaggle.com/jessemostipak/hotel-booking-demand)

### 2. [Web Traffic Forecasting](https://github.com/benyamindariadi/machine-learning/tree/master/web-traffic-timeseries)

- **Context:** This data set contains web traffic time series in median of 7h, 28h ,and etc.
- **Project:** Predict the number of Web traffic.
- **Algorithm:** Recurrent Neural Networks (LSTM).
- **Dataset:** [Source.](https://www.kaggle.com/ymlai87416/wiktraffictimeseriesforecas)

### 3. [Stock Forecasting](https://github.com/benyamindariadi/machine-learning/tree/master/stock-prediction)

- **Context:** This project consumes data from Alpha Vantage API
- **Project:** Predict the stock in the next a couple of days.
- **Algorithm:** Recurrent Neural Networks (LSTM).
- **Dataset:** [Source.](https://www.alphavantage.co/)

## Image

### 1. [LEGO images Classifier](https://github.com/benyamindariadi/machine-learning/tree/master/LEGO%20img%20classifier)

- **Context:** Lego bricks are nice to use to create a recognition model. The model generated by computer rendering. First version of dataset in 2018 using Blender, but decided to create a new version in 2019 using Autodesk Maya 2020.
- **Project:** Make model using CNNs to categorize the type of LEGO. I used image generator in my model because the complete dataset is too large to be processed in my personal computer. I used 12 classes (12 types of LEGO) in this project
- **Algorithm:** Convolutional Neural Networks.
- **Dataset:** [Source.](https://www.kaggle.com/joosthazelzet/lego-brick-images)

### 2. [Four Shapes (Square, Star, Circle, and Triangle)](https://github.com/benyamindariadi/machine-learning/tree/master/Four_Shapes_CNNs_AutoEncoder)

- **Context:** This dataset contains 16,000 images of four shapes; square, star, circle, and triangle. Each image is 200x200 pixels.
- **Project:**
  - Build image classifier model using CNNs.
  - Build denoiser image using AutoEncoder.  
- **Algorithms:** Convolutional Neural Networks, Artificial Neural Networks.
- **Dataset:** [Source.](https://www.kaggle.com/smeschke/four-shapes)

### 3. [Create New Stars!](https://github.com/benyamindariadi/machine-learning/tree/master/Create%20New%20Stars!%20(DCGANs))

- **Context:** I take the dataset from Four Shapes file. The dataset is same, but I only use one file to obtain one single shape which is star.
- **Project:** I made the model to generate "fake" stars images. The model use DC-GANs method to optimize the variety of the result.
- **Algorithm:** Deep Convolutional-Generative Adversarial Network.
- **Dataset:** [Source.](https://www.kaggle.com/smeschke/four-shapes)

### 4. [Rock Paper Scissors Classification](https://github.com/benyamindariadi/machine-learning/tree/master/rock-paper-scissor-classification)

- **Context:** This data set contain datasets of human hand in shape of rock, paper and scissor
- **Project:** Classify whether the input image is rock, paper, or scissor.
- **Algorithm:** CNN.
- **Dataset:** [Source.](https://www.kaggle.com/drgfreeman/rockpaperscissors)

### 5. [Shape Classification (Square, Star, Circle, and Triangle)](https://github.com/benyamindariadi/machine-learning/tree/master/shape-classification)

- **Context:** This project almost the same as the Four shapes project, but there is an addition to convert the model into tensorflow lite
- **Project:**
  - Build image classifier model using CNNs.
  - Convert the model into Tensorflow Lite, so it can be attached to mobile app.
- **Algorithms:** Convolutional Neural Networks
- **Dataset:** [Source.](https://www.kaggle.com/smeschke/four-shapes)

## ComputerVision

### 1. [Where's the ramyun(FLANN kd-tree OpenCV)](https://github.com/benyamindariadi/computer-vision/tree/master/Where's%20the%20ramyun(FLANN%20kd-tree))

- **Project:** This simple code contains object detection using the FLANN kd-tree library. Here will be demonstrated ramyun detection among other instant noodle products.
- **Algorithm:** FLANN kd-tree.

### 2. [Tracking APIs (OpenCV)](https://github.com/benyamindariadi/computer-vision/tree/master/Tracking%20APIs%20(OpenCV))

- **Project:** This simple code contains various object detection algorithms in opencv.
- **Algorithms:**
  - TrackerBoosting
  - TrackerMIL
  - TrackerKCF
  - TrackerTLD
  - TrackerMedianFlow

### 3. [Object detection (YOLOv3)](https://github.com/benyamindariadi/computer-vision/tree/master/Object%20detection%20(YOLOv3))

- **Project:** This simple code is demo of object detection code using the YOLO version 3 algorithm. This code uses a pre-trained YOLOv3 model.
- **Pre-trained model:** @article{YOLOv3, title={YOLOv3: An Incremental Improvement}, author={J Redmon, A Farhadi }, year={2018}

## Text

### 1. [Gender Prediction (Bidirectional RNN-GloVe)](https://github.com/benyamindariadi/NLP/tree/master/Gender%20Prediction%20(Bidirectional%20RNN-GloVe))

- **Context:** This model is created to predict the gender of Twitter users based on tweets and profile descriptions. The model uses GloVe pre-trained word vectors and GRU bidirectional architecture. In the notebook made several models with reference to tweets, profile descriptions and both (multi-inputs) to see the comparison of the model's performances.
- **Project:** Predict the number of booking.
- **Algorithm:** Bidirectional Recurrent Neural Networks (GRU).
- **Dataset:** [Source.](https://www.kaggle.com/crowdflower/twitter-user-gender-classification)

### 2. [ChatBot-TRANSFORMER](https://github.com/benyamindariadi/machine-learning/tree/master/ChatBot-TRANSFORMER)

- **Project:** Create a ChatBot
- **Algorithm:** [Transformer.](https://arxiv.org/abs/1706.03762)
- **Dataset:** [Source.](https://www.cs.cornell.edu/~cristian/Cornell_Movie-Dialogs_Corpus.html)

## Contributing

Please read [CONTRIBUTING.md](CONTRIBUTING.md) for details on our code of conduct, and the process for submitting pull requests to us.

## Contributors

[Benyamin Dariadi](https://github.com/benyamindariadi)

[I Gusti Bagus A](https://github.com/rainoverme002)

See also the list of [contributors](CONTRIBUTORS.md) who participated in this project.

## License

This project is licensed under the MIT License - see the [LICENSE.md](LICENSE.md) file for details

## Acknowledgments

* Thanks for all of the contributions to this project

## Quotes of the Day

* Stay learning and you will get what you want

This repository contains portfolio of data science projects completed by me for academic, self learning, and hobby purposes.
The projects are presented in the form of Jupyter Notebooks.

[한국어 포트폴리오](https://github.com/RangDuk/data-science-portfolio/blob/master/README(kr).md)는 여기에 있습니다.

## Content

### Kaggle Competitions
* [Home Credit Default Risk by Home Credit Group](https://www.kaggle.com/c/home-credit-default-risk)
    * [EDA and Random Tree baseline model](https://github.com/RangDuk/data-science-portfolio/blob/master/201808%20-%20Home%20Credit%20Default%20Risk/Home%20Credit%20Default%20Risk%20-%20EDA.ipynb) - Did extensive EDA on the datasets and feature importance explorations. Also built baseline Random Tree model (Score: 0.731).
    * [Feature Enginnering](https://github.com/RangDuk/data-science-portfolio/blob/master/201808%20-%20Home%20Credit%20Default%20Risk/Home%20Credit%20Default%20Risk%20-%20Feature%20Engineering.ipynb) - Manual feature engineer process. Aggregated features were created. (interaction features/ indicator features should be added in the future) Feature selections included.
    * [LightGBM](https://github.com/RangDuk/data-science-portfolio/blob/master/201808%20-%20Home%20Credit%20Default%20Risk/Home%20Credit%20Default%20Risk%20-%20LightGBM.ipynb) - Used LightGBM to make prediction. Hyperparameter tuning included.

### Deep Learning
* [Multi-Layer Perceptron: MNIST Data](https://github.com/RangDuk/data-science-portfolio/blob/master/Multi-Layer%20Perceptron%20-%20MNIST%20data.ipynb) - Using multi-layered perceptron method to classify image files. Utilized Adam Optimizer to minimize cost.  
* [Deep Neural Network Classifier: Fake Bank Note Authentication](https://github.com/RangDuk/data-science-portfolio/blob/master/Deep%20Neural%20Network%20Classifier%20-%20Fake%20Bank%20Note%20Authentication.ipynb) - Using DNNClassifier method to predict whether the bank note is authentic or not. Used StandardScaler() to normalize the data and then DNN Estimator to predict the outcome. The accuracy was unusually high, so performed Random Forest Classifer to do a reality check.
* [Facial Expression Recognition](https://github.com/RangDuk/data-science-portfolio/blob/master/Facial%20Expression%20Recognition.ipynb) - Only used Numpy to build one hidden-layered neural network model. Purpose of this project was to deepen understanding of the structure of neural network and to unveil what is happening under the hood of modern neural network libraries.
*tools used: Tensorflow, scikit-learn, Pandas, Seaborn, Matplotlib*

### Machine Learning
* [Logistic Regression: Advertisement Clicked](https://github.com/RangDuk/data-science-portfolio/blob/master/Logistic%20Regression%20-%20Is%20the%20Advertisement%20Clicked.ipynb) - Using logistic regression model to predict whether the user has clicked on the internet advertisement.
* [K-Nearest Neighbors: Unsupervised Machine Learning](https://github.com/RangDuk/data-science-portfolio/blob/master/K%20Nearest%20Neighbors%20-%20Classified%20Dataset.ipynb) - Using K Nearest Neightbors method to build prediction model. The learner is unsupervised, data is classified.
* [Decision Tree and Random Forest: Lending Club valuation](https://github.com/RangDuk/data-science-portfolio/blob/master/Decision%20Trees%20and%20Random%20Forest%20-%20'Who%20wants%20my%20money'%20%20Lending%20Club.ipynb) - Using both Decision Tree and Random Forest model to predict whether borrowers will pay back in time. Used real-life data from [Lending Club](https://www.lendingclub.com/info/download-data.action).
* [K Means Clustering: Public and Private Universities Cancer](https://github.com/RangDuk/data-science-portfolio/blob/master/K%20Means%20Clustering%20Project%20.ipynb) - Using K Means Clustering method to cluster universities into two groups, Private and Public.
* [Recommender Systems: Movie Lens Data](https://github.com/RangDuk/data-science-portfolio/blob/master/Recommender%20Systems%20-%20Collaborative%20Filtering%20on%20Movie%20Lens%20Data%20Set.ipynb) - Built recommender systems model using various Collborative Filtering methods. Memory-based collaborative filtering by computing cosine similarity of users and items. Model-based collboartive filtering by using singular value decomposition (SVD).
* [Support Vector Machines - Iris Flowers](https://github.com/RangDuk/data-science-portfolio/blob/master/Support%20Vector%20Machines%20-%20Iris%20Flower%20Data%20Set.ipynb) - Using support vector machine process to predict the species of iris flower in the dataset. Also used **gridsearch** to tune the parameters of the model.
*tools used: scikit-learn, Pandas, Seaborn, Matplotlib*

### Natural Language Processing
* [Natural Language Processing: Study of Yelp Review](https://github.com/RangDuk/data-science-portfolio/blob/master/NLP%20-%20Yelp%20Review.ipynb) - Using CountVectorizer, Naive Bayes model, and built pipeline to predict number of stars given by analyzing the text reviews on Yelp.
*tools used: NLTK, scikit*

### Data Analysis and Visualization
* [EDA: 911 Calls from Montegomery County](https://github.com/RangDuk/data-science-portfolio/blob/master/EDA%20-%20911%20Calls%20from%20Montgomery%20County.ipynb) - EDA on 911 Calls from Montegomery County. Used all the basic visualization libraries and Follium to map the strange occurance of 911 Calls.
* [EDA: Study of Bank Stock Price](https://github.com/RangDuk/data-science-portfolio/blob/master/EDA%20-%20Bank%20Stock%20Price.ipynb) -EDA on bank stock prices from 2006 to 2015. Broke down to explain different incidents that had effects on stock prices.
*tools used: Pandas, Folium, Seaborn, Matplotlib, Plotly.js*

### Others
* [Principal Component Analysis](https://github.com/RangDuk/data-science-portfolio/blob/master/Principal%20Component%20Analysis.ipynb) - Practices using PCA to transform the data. Data used is the breast cancer dataset in sklearn.datasets.
* [Singular Value Decomposition](https://github.com/RangDuk/data-science-portfolio/blob/master/Recommender%20Systems%20-%20Collaborative%20Filtering%20on%20Movie%20Lens%20Data%20Set.ipynb) - SVD example is present in the Recommender System (Model-Base Collborative Filtering Method)

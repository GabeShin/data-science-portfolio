이 리포지토리는 데이터 사이언스 포트폴리오용 프로젝트 모음집입니다.
모든 프로젝트들은 Jupyter Notebook을 이용하여 작성하였습니다.

There is [English](https://github.com/RangDuk/data-science-portfolio/blob/master/README.md) versions of the portfolio.

## Content

### Kaggle Competitions
* [Home Credit Default Risk by Home Credit Group](https://www.kaggle.com/c/home-credit-default-risk)
    * [EDA and Random Tree baseline model](https://github.com/RangDuk/data-science-portfolio/blob/master/201808%20-%20Home%20Credit%20Default%20Risk/Home%20Credit%20Default%20Risk%20-%20EDA.ipynb) - 강도 높은 EDA. 베이스라인 모델으로 사용할 Random Tree Model 시도 (점수: 0.731). Random Tree Model을 통해 특징성 강한 feature들 선별.
    * [Feature Enginnering]() - 곧 업데이트 예정

### Deep Learning
* [Facial Expression Recognition](https://github.com/RangDuk/data-science-portfolio/blob/master/Facial%20Expression%20Recognition.ipynb) - Numpy 라이브러리만 사용하여 인공신경망 모델을 만듬. 이 프로젝트는 본인의 인공신경망 모델에 대한 이해도를 높이고 단순 plug-and-run이 아닌 모던 라이브러리에 대한 이해도를 높이기 위함. Tanh Activation을 이용한 binary classification 모델과 Softmax를 사용한 classification 모델 만듬.
* [Multi-Layer Perceptron: MNIST Data](https://github.com/RangDuk/data-science-portfolio/blob/master/Multi-Layer%20Perceptron%20-%20MNIST%20data.ipynb) - 다층 퍼셉트론을 이용한 딥러닝을 통해 MNIST 이미지 파일을 예측함. Adam Optimizer를 활용.  
* [Deep Neural Network Classifier: Fake Bank Note Authentication](https://github.com/RangDuk/data-science-portfolio/blob/master/Deep%20Neural%20Network%20Classifier%20-%20Fake%20Bank%20Note%20Authentication.ipynb) - DDNClassifer을 이용한 딥러닝을 활용하여 은행권(bank note)이 진짜인지 가짜인지 구분하는 모델 개발. 예측지수가 예상외로 높아 현실성검사를 위해 마지막에 랜덤 포레스트 모델로 검증.
*tools used: Tensorflow, scikit-learn, Pandas, Seaborn, Matplotlib*

### Machine Learning
* [로지스틱 회귀분석: Advertisement Clicked](https://github.com/RangDuk/data-science-portfolio/blob/master/Logistic%20Regression%20-%20Is%20the%20Advertisement%20Clicked.ipynb) - 로지스틱 회귀분석을 사용해 사용자가 인터넷 광고에 클릭을 할지 예측하는 예측모델.
* [K-최근접 이웃기법: Unsupervised Machine Learning](https://github.com/RangDuk/data-science-portfolio/blob/master/K%20Nearest%20Neighbors%20-%20Classified%20Dataset.ipynb) - K-최근접 이웃기법을 이용한 비지도학습 시스템 예측모델 개발.
* [디시전 트리와 랜덤 포레스트: Lending Club valuation](https://github.com/RangDuk/data-science-portfolio/blob/master/Decision%20Trees%20and%20Random%20Forest%20-%20'Who%20wants%20my%20money'%20%20Lending%20Club.ipynb) - 디시전 트리와 랜덤 포레스트 모델을 사용하여 대출자가 제시간에 돈을 갚을지 예측하는 모델.[Lending Club](https://www.lendingclub.com/info/download-data.action)에서 실제 데이터 수집하여 이용.
* [K-평균 군집화: Public and Private Universities Cancer](https://github.com/RangDuk/data-science-portfolio/blob/master/K%20Means%20Clustering%20Project%20.ipynb) - K-평균 군집화를 이용하여 대학교를 공립/사립으로 분류하는 모델.
* [추천자 시스템: Movie Lens Data](https://github.com/RangDuk/data-science-portfolio/blob/master/Recommender%20Systems%20-%20Collaborative%20Filtering%20on%20Movie%20Lens%20Data%20Set.ipynb) - 총 세가지의 협업 필터링 알고리즘을 사용하여 추천자 시스템 개발. 메모리기반 협업 필터링을 통해 유저 사이(1)의/ 영화 사이(2)의 코사인 유사도 계산비교 모델. SVD 분해기법을 이용한 모델기반 협업 필터링(3) 모델.
* [서포트 벡터 머신 - Iris Flowers](https://github.com/RangDuk/data-science-portfolio/blob/master/Support%20Vector%20Machines%20-%20Iris%20Flower%20Data%20Set.ipynb) - 서포트 벡터 머신 알고리즘을 사용하여 아이리스의 종류를 구분하는 모델 개발.또한 **그리드서치** 를 사용하여 모델 파라미터를 최적화.
*tools used: scikit-learn, Pandas, Seaborn, Matplotlib*

### Natural Language Processing
* [자연화처리: Study of Yelp Review](https://github.com/RangDuk/data-science-portfolio/blob/master/NLP%20-%20Yelp%20Review.ipynb) - CountVectorizer, NLPK, Naives Bayes Model, 그리고 데이터 파이프라인 프로세스를 사용하여 Yelp에 남겨진 텍스트 리뷰를 이용한 별점 예측 모델.
*tools used: NLTK, scikit*

### Data Analysis and Visualization
* [데이터 분석, 시각화, 맵핑: 911 Calls from Montegomery County](https://github.com/RangDuk/data-science-portfolio/blob/master/EDA%20-%20911%20Calls%20from%20Montgomery%20County.ipynb) - Montegomery County에서 응급실로 온 연락 데이터 분석 및 시각화. 각종 데이터분석 라이브러리 이용, 또한 Follium라이브러리를 이용한 데이터 지도화.
* [데이터 분석 및 시각화: Study of Bank Stock Price](https://github.com/RangDuk/data-science-portfolio/blob/master/EDA%20-%20Bank%20Stock%20Price.ipynb) - 2006부터 2015년 까지의 뉴욕 은행 주식가 데이터 분석. 각종 주식 변동 확인 후 이유 및 영향 분석.
*tools used: Pandas, Folium, Seaborn, Matplotlib, Plotly.js*

### Others
* [주성분 분석 ](https://github.com/RangDuk/data-science-portfolio/blob/master/Principal%20Component%20Analysis.ipynb) - 주성분 분석을 이용하여 데이터 요약 연습. 유방암 데이터를 사용하여 데이터 요약 후 시각화.
* [특이값 분해 (SVD) ](https://github.com/RangDuk/data-science-portfolio/blob/master/Recommender%20Systems%20-%20Collaborative%20Filtering%20on%20Movie%20Lens%20Data%20Set.ipynb) - 특이값 분해 연습은 추천자 시스템에서 사용. (모델기반 협업 필터링)

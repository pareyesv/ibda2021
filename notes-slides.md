---
title: Practical Data Analytics for Solving Real World Problems - participants
author:
- Maria Teresa Grifa
- Patricio Reyes
date: 2021-01-26
slideOptions:
    transition: 'slide'
---

# Hello :hand:

[![hackmd-github-sync-badge](https://hackmd.io/rd7ivXfwTUOY5Jl1hw7D7A/badge)](https://hackmd.io/rd7ivXfwTUOY5Jl1hw7D7A)


<!-- .slide: data-background="https://media.giphy.com/media/26xBwdIuRJiAIqHwA/giphy.gif" -->

----

## Maria Teresa Grifa

- data scientist at Bridgestone EMA
- PhD candidate at University of L'Aquila
- github: MT-G

----

## Patricio Reyes

- researcher at BSC, [Data&Vis Team](http://bsc.es/viz/)
- member of PyBCN 
- repos
    - [cuentalo](https://github.com/BSCCNS/cuentalo-dataset)
    - [Landscape Steiner](https://github.com/pareyesv/landscape_steiner) ==looking for collaborators==
- twitter: @pareyesv
- github: [pareyesv](https://github.com/pareyesv)


----

# ==You ?==

----

### Aude Carreric

- researcher at BSC
- twitter: @aude_carreric
- github: acarreri 

----

## Dadaist approach

feel free to collaborate on this presentation
- suggestions?
- new content?
- errors?
    - typos
    - Italenglish?
    - Spanglish?

----

# Share your roadmap

----

## ==Wise Apple Bowl 2020==

- Reading Group [Fluent in Python](https://github.com/fluentpython)
- [Landscape Steiner Project](https://github.com/pareyesv/landscape_steiner)
- Reading Group [Elements of Statistical learning](https://web.stanford.edu/~hastie/ElemStatLearn) 
- [Alice in Wonderland: Object Oriented Programming in Lewis Carroll Games](https://github.com/MT-G/OOP-Games)
- [PyDay  BCN 2020](https://pybcn.org/events/pyday_bcn/pyday_bcn_2020/)
- [Advent of Code 2020](https://adventofcode.com/)

----

## A different approach
  
- [Awesome Python Features Explained Using the World of Magic](https://github.com/zotroneneis/magical_universe)

----

## Let's collaborate

- join Slack
    - patcibda2021.slack.com
- introduce yourself to your group

----

## Next steps

- share yor notes [Example: fastai course](https://becominghuman.ai/fast-ai-v3-2019-lesson-1-image-classification-db93bb63e819)
- share your ML roadmap [Example](https://github.com/mrdbourke/machine-learning-roadmap)
- start a repository
    - wiki?
    - README.md
    - tools
        - markdown
- show your own data science roadmap

---

<!-- .slide: data-background="https://media.giphy.com/media/LQiq27myXGPXO6WzAE/giphy.gif" -->

# Pre-requisites

----

- Python
- Chocolate :chocolate_bar: 
- Anaconda 
- Brewed coffee :coffee:
- Good Will!
- Carbonara 

---

# How to structure a data science project

----

## collaboration :handshake:

- slack is not enough!
- team, team, team

----

## 1. "I work alone. I don't care" :sunglasses: 

- you collaborate with yourself
- your _future self_ will need
    - documentation
    - debugging

----

## you always need to ==collaborate==

- looking for advice
    - blogs: bloggers share knowledge
    - books: authors share knowledge
- beta-testing
    - you are not the best programmer to test your own code
- why don't you ask for collaboration?!?!

----

## 2. "I work on a team"

- your _future self_ is part of the same team
    - smart member :nerd_face: 
    - your _future self_ will need documentation
- you (your code) have to interact with others
    - documentation
    - README file
    - end-user
        - how to run the code?
    - developer
        - how to start working on the code?

----

## reproducibility

- [Reproducibility PI Manifesto :: Lorena A. Barba Group](https://lorenabarba.com/gallery/reproducibility-pi-manifesto/)
- [slides](https://speakerdeck.com/labarba/how-to-run-a-lab-for-reproducible-research)

----

## Data Science Template

- [Cookiecutter Data Science](https://drivendata.github.io/cookiecutter-data-science/)
    - [You will thank you](https://drivendata.github.io/cookiecutter-data-science/#:~:text=You%20will%20thank%20you)

----

## Install the template

- [data science template](https://drivendata.github.io/cookiecutter-data-science/#example)

----

## Directory Structure

- [structure](https://drivendata.github.io/cookiecutter-data-science/#directory-structure)

----

## Data is immutable

- ==Don't ever edit your raw data==
    - especially not manually...
    - ...and especially not in Excel 
- ==Don't overwrite your raw data==
- Treat the data (and its format) as immutable.
- data folder in `.gitignore`

----

## Data version control

- databases
- [Data Version Control · DVC](https://dvc.org/)
- [Git Large File Storage](https://git-lfs.github.com)
- [AWS Amazon Simple Storage Service (S3)](https://aws.amazon.com/s3/)

----

- Data science project template
    - templates
    - documentation
    - README files
    - LICENSE
    - semantic versioning
    - collaboration
        - issues
        - Pull Requests

----

- tools
    - Github
    - CookieCutter
        - See also: [Copier](https://github.com/copier-org/copier)
    - Sphinx

---

# Data Analytics Tools

----

## jupyter notebook

- [example: data analysis notebook](https://github.com/pareyesv/ibda2021/blob/main/notebooks/data-analysis/1.1-data_analysis.ipynb) :thumbsup:

----

## Domain Knowledge

- Data (Big Data?)
    - Velocity (speed at which data is generated, collected, analyzed)
    - Volume (is data stored in distributed systems?)
    - Variety (different types: structured vs unstructured)
    - Value (worth of data)
    - Veracity

----

## RoadMap

- [Data Preparation](#Data-Preparation)
- [Exploratory Data Analysis](#Exploratory-Data-Analysis)
- [Visualization](#Visualization)
- [Features Engeneering](#Features-Engeneering)
- [Features Selection](#Features-Engeneering)
- [Modelling](#Modelling)
- [Deployment](#Deployment)

----

## Data Preparation

----

### Raw Data

- structured data: data matrix
- graph: web and social networks
- spatial data
- time series 
    - sensors data
    - stock exchange data
- unstructured data: 
    - text
    - images

----

### Clean Data

- extract data with different formats
    - excel, json, csv, pdf, jpg, mp4, etc.
- evaluation of data accuracy and reliability
    - presence of missing values
    - outliers
    - inconsitencies
    - level of noise

----

<!-- .slide: data-background="https://i2.wp.com/flowingdata.com/wp-content/uploads/2014/09/outlier.gif" -->

----

### Data Consolidation

- consistency across different data sources
- consistency of units
- consistency of scales
- consistency of file, folder names, etc..

----

## Exploratory Data Analysis

----

### Problem identification

- analysis before modelling
- the objective is to understand the problem in order to generate testable hypotheses
- clear, concise and measurable
- define 
    - target/label (dependent variable)
    - features (indepent variables)
- crucial to select the right class of algorithms

----

### Basic Statistics

- describe dimensions
- type of distributions
- descriptive statistics
    - mean, median, mode, std
- correlation between features
- relationships and pattern due to the structure of the data

----

![duck](https://flowingdata.com/wp-content/uploads/2014/06/rabbit-or-duck.jpg "Duck") 

----

## Visualization

==(Check the course on Friday!)==

![image alt](https://totaldatascience.com/wp-content/uploads/2019/10/p66.jpg)

----

### Visualization packages

- matplotlib
- plotly
- seaborn
- bokeh
- altair

----

### Quintessential rules to use in visualization

- less is more 
    - be clear 
    - check properly the type of graph

----

- reduce the clutter
    - avoid unecessary or distracting visual elements
        - ornametal shading
        - dark gridlines
        - 3D when not mandatory
        - thick marks
- [preattentive visual processing](
https://www.thinkingondata.com/something-about-viridis-library/)
- [overview from kaggle](https://www.kaggle.com/python10pm/plotting-with-python-learn-80-plots-step-by-step)

----

![data vis or modern art](https://venngage-wordpress.s3.amazonaws.com/uploads/2020/06/image17.png "data vis or modern art")

----

## Feature Engineering

----

### Features creations 
    
- domain knowledge
    - brainstorming
- dataset based
    - sum, product, linear combination, power, etc..
    - lags in time
    - computation of physical quantities
- join/aggregate
- smoothing
- encoding (categorical data)

----

### Features Check

- imbalanced datasets
- duplicated data

**Pass go and visualize**

![image alt](https://boardgamedesigns.com/wp-content/uploads/2017/12/go-to.png "go")

----

### Split Data

divide date into training set and testing set
divide training set into training set ad validation set
- training dataset
    - used to fit the machine learning model
- validation dataset
    - used to find best features, check model performace, tuning
- testing dataset
    - used to evaluate the fit machine learning model

----

### Splitting Techniques

- basic split
    - 60% test, 40% train
- cross validation
    - k-fold cross validation 
    - rolling cross validation (time series)
    - stratified cross validation

----

### Model Selection

#### supervised learning
- use training samples with known classes to classify new data
- we are given examples of inputs and associated outputs
- we learn the relationship between them

#### unsupervised learning
- training samples have no class information
- we are given inputs but no outputs 
- we learn the _latent_ labels

----

### Performance Metric Classification

**classification metrics example**
- classification accuracy: 
ratio number of correct predictions on all predictions made
- confusion matrix
table presents predictions on the x-axis and accuracy outcomes on the y-axis

----

### Performance Metric Regression

**regression metrics examples**
- mean squared error
like the mean absolute error in that it provides a gross idea of the magnitude of error
- R squared 
coefficient of determination
indication of the goodness of fit of a set of predictions to the actual values
value between 0 and 1 for no-fit and perfect fit respectively.

----

## Machine Learning Map

https://scikit-learn.org/stable/tutorial/machine_learning_map/index.html

----

## Scikit-Learn

https://scikit-learn.org/stable/

----

## R-Squared feat. Regression

- Output  Y, with observed data points
- Simple way of predicting future observations isnthe **mean of our existing observations**

![mean y](https://miro.medium.com/max/3000/1*mljojvPo_QzbmEaAVUmtnQ.png =300x300)

----

To improve 
- linear regression model f, to predict values of Y

![f](https://miro.medium.com/max/3000/1*qGZvdV4TSuYmTt27s7SAjA.png =300x300)

----

- How good our model is?
- How much of an improvement is our model f over the _baseline mean model_?
- _How much of the variance from the mean does our model account for?_
   

----

- A regression is a model that minimises the difference between the model’s predicted values of y, and the actual observed values of y (aka residuals)

![residuals](https://miro.medium.com/max/3000/1*nOE9bCbSwXz3vaRzM0r-zw.png =400x200)

----

- Summing up the residuals...
- ...ending up to sum their squares

![squared res](https://miro.medium.com/max/875/1*F3l0a4hMQPo_eOmGZMRE-w.png =500x300)

----

- the blue squares are another way of visualising the part of the error our model can not explain
- R-squared: shred variance that is explained by the model

$$R^{2} = 1 - \frac{Sum(\mbox{blue squares})}{Sum(\mbox{orange squares})}$$

$$R^{2} =
\begin{cases}
1 \quad  \mbox{perfectly explains data}\\
0 \quad \mbox{explains precisely nothing}
\end{cases}
$$

----

### Define Benchmark Model

- use a simple model 
- your final model have to beat the benchmark
- benchmark should be better than random

**Pass go and check another algo**
![image alt](https://boardgamedesigns.com/wp-content/uploads/2017/12/go-to.png "go")

----

### Know the capabilities of algorithms

- NaN handling

### Compare more than one algorithm

**Pass go and check another algo**
![image alt](https://boardgamedesigns.com/wp-content/uploads/2017/12/go-to.png "go")

----

### Check Model Performace 

**overfitting**
- models the training data too well
- learns detail and noise in the training data and it negatively impacts the performance of the model on new data
- probs: the models ability of the model to generalize
- when:
    - nonlinear model, flexibility when learning a target function

----

**underfitting**
- can neither model the training data nor generalize to new data
- probs: poor performance on the training data

----

### Model Performace Visualization

![image alt](https://cdn.hashnode.com/res/hashnode/image/upload/v1591931791416/qtb6eievP.png)

----

![overfitting](https://img-9gag-fun.9cache.com/photo/aOYo6Ly_700bwp.webp =400x400)

----

### Improve Model Performace

**hyperparameter tuning**
model configuration argument specified by the developer to guide the learning process for a specific dataset

- grid search:
define a search space as a grid of hyperparameter values and evaluate every position in the grid

----

![xkcd](https://imgs.xkcd.com/comics/machine_learning_2x.png "xkcd" =400x400)

----

## Deployment

- according to [wikipedia](https://en.wikipedia/wiki/Software_deployment)
    > Software deployment is all of the activities that make a software system available for use.

----

> notebooks are just for exploration

- well...
- let's deploy jupyter notebooks :zipper_mouth_face: 

----

### ==papermill== + nbconvert

- run notebooks from command line
    - parameterize
        - from command line
        - from yaml config file
    - inject variables into the notebook
        - [cell tagged `parameters`](https://papermill.readthedocs.io/en/latest/usage-parameterize.html)
- See [how Netflix uses papermill](https://netflixtechblog.com/scheduling-notebooks-348e6c14cfd6)

----

### papermill + ==nbconvert==

- jupyter notebook $\rightarrow$ webpage (html)
- [how to](https://nbconvert.readthedocs.io/en/latest/usage.html)

----

- further reading:
    - [Automated Report Generation with Papermill: Part 1 - Practical Business Python](https://pbpython.com/papermil-rclone-report-1.html) 
    - [Automated Report Generation with Papermill: Part 2 - Practical Business Python](https://pbpython.com/papermil-rclone-report-2.html)

- cons
    - nbconvert
        - no interactivity
        - javascript running in the browser

----

### webapps

- voilà
    - [voilà: notebook running on heroku](https://pythonforundergradengineers.com/deploy-jupyter-notebook-voila-heroku.html)
- streamlit
    - [GitHub - alonsosilvaallende/streamlit-test](https://github.com/alonsosilvaallende/streamlit-test)
- anvil
    - [local notebook to a webapp](https://anvil.works/learn/tutorials/jupyter-notebook-to-web-app)
        - [simple tutorial](https://medium.com/datadriveninvestor/create-your-own-machine-learning-app-with-anvil-basic-6bf3503e80f1)
    - [webapp with user registration](https://anvil.works/learn/tutorials/hello-world)

---

# Hands-on

----

## 1. exploratory data analysis

<!-- .slide: data-background="https://media.giphy.com/media/OJw4CDbtu0jde/giphy.gif" -->

[colab](https://colab.research.google.com/github/pareyesv/ibda2021/blob/main/notebooks/hands-on/1.1-eda_pandas_profiling.ipynb) :arrow_left: 

- [pandas-profiling](https://pandas-profiling.github.io/pandas-profiling/docs/master/rtd/)
    - [Titanic](https://pandas-profiling.github.io/pandas-profiling/examples/master/titanic/titanic_report.html)
    - [1978 Automobile dataset](https://pandas-profiling.github.io/pandas-profiling/examples/master/stata_auto/stata_auto_report.html)
    - [Census Dataset](https://pandas-profiling.github.io/pandas-profiling/examples/master/census/census_report.html)
    - [more examples](https://github.com/pandas-profiling/pandas-profiling#examples)

----

## 2. scikit-learn (Colab)

[colab](https://colab.research.google.com/github/pareyesv/ibda2021/blob/main/notebooks/hands-on/1.2-plot_transformed_target.ipynb) :arrow_left: 

- linear regression
    - evaluation with and without trasformation
- decision tree
    - evaluation with and without trasformation

----

## 3. advent of code 2020 

[notebook colab](https://colab.research.google.com/github/pareyesv/ibda2021/blob/main/notebooks/hands-on/1.3-advent_of_code_2020_day1.ipynb) :arrow_left: 

- Day 1

----

## project template

- [Cookiecutter data science](https://drivendata.github.io/cookiecutter-data-science/)

---

<!-- .slide: data-background="https://media.giphy.com/media/DAtJCG1t3im1G/giphy.gif" -->


# Thanks!

See you on slack!

---

{%youtube HJog7PfkNRY %}

---

# Tips

----

- Data Consolidation
    - [great-expectation](https://greatexpectations.io/)
- EDA 
    - [pandas profiling](https://github.com/pandas-profiling/pandas-profiling)
- Visualization 
    - [understand types of plots](https://totaldatascience.com/wp-content/uploads/2019/10/p75.png)
    - [matplotlib cheatsheet](https://totaldatascience.com/wp-content/uploads/2019/10/p59.pdf)
- Choosing the right estimator, from [scikit-learn](https://scikit-learn.org/stable/tutorial/machine_learning_map/index.html)
- RISE slides in jupyter notebook

---

# Acknowledgements

---

- a BIG thanks to [José Carlos Carrasco Jimenez](https://www.bsc.es/carrasco-jimenez-jose-carlos)
- [CINECA course](https://eventi.cineca.it/en/hpc/school-scientific-data-analytics-and-deep-learning-0) 
    - check this out! 

----

## Thanks to all the contributors

- Maria Teresa Grifa
- Patricio Reyes
- [Aude Carreric](https://twitter.com/aude_carreric)

----

# References

----

- learning
    - [Machine Learning Mastery's FAQ](https://machinelearningmastery.com/faq/)
        - How to start in Machine Learning?
        - Do I need a degree?
- tutorials
    - [scikit-learn](https://scikit-learn.org/stable/)
    - [Machine Learning Mastery](https://machinelearningmastery.com)
- books
    - [Fluent in Python](https://github.com/fluentpython)
        - Luciano Ramhamallo
    - [The Elements of Statistical Learning](https://web.stanford.edu/~hastie/ElemStatLearn) 
        - J.H. Friedman, R. Tibshirani, T.Hastie



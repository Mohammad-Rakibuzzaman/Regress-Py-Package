# Regression Evaluating Package

[![image](https://img.shields.io/pypi/v/py-package-template.svg)](#)
[![Build Status](https://travis-ci.org/AlexIoannides/py-package-template.svg?branch=master)](#)

The regress_py package allows users to download the contents of this [GiHub repository](https://github.com/roman1117459/regress_py_package), containing a Regression Evaluating Python package project to be used as a gradient descent, loss and regression problem for kick-starting Machine learning or Deep Learning algorithm model evaluation of **any** type of Package; destined for upload to just for local install using Pip. The downloaded package includes the following components to measure or evalute the loss or any kind of measurement without having to spend time cloning existing set-ups from other projects:

- a minimal `setup.py` file;
- testing with PyTest;
- Evaluate the gradient descent loss and info;

A description of how to work with (and modify) each of these components, is provided in more detail in the sections that follow-on below, as well as in the documentation and within the example code bundled with the package.

This is obviously a opinionated view of how a Python package project ought to be structured, based largely on my own experiences and requirements.

## Downloading a regress_py package

To down load the latest version of this project located in [this GiHub repository](https://github.com/roman1117459/regress_py_package), execute the following command from the command line:

```bash
pip install regresspy
```

This will be downloaded to the current directory and will contain the following directory structure:

```bash
regress_py_package/

 |-- regresspy/
 |-- |-- __init__.py
 |-- |-- gradient_descent.py
 |-- |-- loss.py
 |-- |-- regression.py
 |-- test/
 |-- |   __init__.py
 |-- |   model.py
 |-- |   test.py
 |-- README.md
 |-- requirements.txt
 |-- setup.py
```

## Installing

Install and update using [pip](https://pip.pypa.io/en/stable/quickstart/):

```bash
pip install wheel
```

Wheel will build the package. and after that run:

```bash
python setup.py bdist_wheel
python setup.py sdist bdist_wheel

```

It will create 2 new directory which is called build and dist. Inside of dist folder there will found a wheel file. After run this command:

```bash
pip install .\regresspy-0.2.0-py3-none-any.whl
```

## Test the regress_py_package

You can also check the liner regression trained rmse value from scikitlearn and also can observe loss values according to the epochs. You can also check your train rmse according to my code. To check the model.py, open powershell and run this code:

```bash
python model.py
```

You can also test different kind of loss such as mae, sse, mse, rmse. For this just run the below code:

```bash
python test_loss.py
```

Steps :

-First of all I have set weight and bias in predict function and return prediction.

-Then I have measure the score all the (Rmse, MSE, mae, sse) function using if, else from the score function and return the score.

-Actually i change self weight and bias but in the initialize_weights function but my course teacher already did it.

-And finally I have change the train function using forward and backward function from the gradient descent file.

---

## Model.py file.

-At first i have changed the reshape in Y predict portion. I got error and again i set the reshape and run successfully.

-I Used the stochastic gradient descent regressor and set iteration 20.

-I trained X, Y through fit function from regression file of the stochastic gradient descent.

-I predict stochastic gradient descent through predict function from regression.py file.

-I calculated stochastic gradient descent rmse through rmse function from loss.py file.

-And finally i got stochastic gradient descent rmse value, where the value is 0.5952712980765811.

-I evaluated regression value through regression function from regression.py file and set epoch/iteration 100 and also learning rate was 0.0001.

-I trained X, Y through fit function from regression file of the regression value.

-I predict regression value through predict function from regression.py file.

-I calculated score of regression rmse through score function from regression.py file.

-And finally i got RMSE value, where the value is 1.2478209695866114

Test :

- Go to the root folder of regresspy and run the command 'pytest' from your command prompt (you should have the pytest module installed on your system)

```bash
pip install pytest
```

-This will automatically check whether your test functions are correct or not.

If 4 item passed correctly. Done! regresspy package is working now. Enjoy......

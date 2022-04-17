
# Overview

If I could predict the stock market with even as much as 30% error, I'd be rich enough to inherit the entire nation of Wakanda, vibranium mountain and all.

There's a reason why an entire district of New York City is dedicated to the art of taming the stock market. Whoever cracks the code even slightly has the potential to get absurd levels of returns, and with those returns, lots of bills.

I'll admit the title of this article is a bit misleading. There is no current method to even remotely accurately predict the stock market, but that's kind of the point.

## Talk about market efficiency and the paradox of passive investors and active investors

## Table of Contents


*   Predicting the market with Stocker
*   Predicting the market with linear regression
*   Predicting the market with neural networks
*   Conclusion -> it's safer to invest in ETFs







# The Stocker Module


```
!git clone https://github.com/WillKoehrsen/Data-Analysis.git
```


```
import os
```


```

!ls
```

    Data-Analysis  gdrive  sample_data



```
os.chdir('Data-Analysis')
```


```
!ls
```

     additive_models	    learning_skills	      sentdex_data_analysis
    'bayesian drake equation'   LICENSE		      setup
     bayesian_log_reg	    logistic_regression       slack_interaction
     bayesian_lr		    medium		      sp500tickers.pickle
     copernican		    nyc_traffic_data	      statistical_significance
     cyclical-features	    over_vs_under	      statistics
     data-science-tips	    pairplots		      stocker
     datashader-work	    plotly		      think_complexity
     distributions		    poisson		      time_features
     ecdf			    prediction-intervals      univariate_dist
     economics		    random_forest_explained   web_automation
     example_notebook	    README.md		      weighter
     Facts			    recall_precision	      weight_loss_challenge
     geo			    requirements.txt	      widgets



```
os.chdir('stocker')
```


```
!ls
```

     data	  __pycache__			 'Stocker Prediction Usage.ipynb'
     dev	  readme.md			  stocker.py
     images  'Stocker Analysis Usage.ipynb'



```
!pip install quandl
!pip install pytrends
```


```
import stocker
```


```
from stocker import Stocker
```


```
goog = Stocker('GOOGL')
```


```
goog.plot_stock()
```

    Maximum Adj. Close = 1187.56 on 2018-01-26 00:00:00.
    Minimum Adj. Close = 50.16 on 2004-09-03 00:00:00.
    Current Adj. Close = 1006.94 on 2018-03-27 00:00:00.
    



![png](images/Stock%20Prediction%20Project_13_1.png)


If you pay attention, you'll notice that the dates for the Stocker object are not up-to-date. It stops at 2018--3-27. Taking close look at the actual module code, we'll see that the data is taken from Quandl's WIKI exchange. Perhaps the data is not kept up to date? 

We can use Stocker to conduct technical stock analysis, but for now we will focus on being mediums. Stocker uses a package created by Facebook called prophet which is good for additive modeling.


```
model, model_data = goog.create_prophet_model(days=90)
# Let's make some predictions
```

    Predicted Price on 2018-06-25 00:00:00 = $1175.64



![png](images/Stock%20Prediction%20Project_16_1.png)


## Evaluation of Predictions

Now we will test the stocker predictions. We need to create a test set and a training set. We'll have our training set to be 2014-2016, and our test set to be 2017. Let's see how accurate this model is.


```
goog.evaluate_prediction()
```

    
    Prediction Range: 2017-03-27 00:00:00 to 2018-03-27 00:00:00.
    
    Predicted price on 2018-03-24 00:00:00 = $1037.84.
    Actual price on    2018-03-23 00:00:00 = $1026.55.
    
    Average Absolute Error on Training Data = $14.31.
    Average Absolute Error on Testing  Data = $65.52.
    
    When the model predicted an increase, the price increased 58.27% of the time.
    When the model predicted a  decrease, the price decreased  48.18% of the time.
    
    The actual value was within the 80% confidence interval 63.60% of the time.



![png](images/Stock%20Prediction%20Project_18_1.png)


This is absolutely horrible!

We'll try again, and this time we'll adjust some hyperparameters.

## Adjusting the Changepoint Priors



```
# changepoint priors is the list of changepoints to evaluate

goog.changepoint_prior_analysis(changepoint_priors=[0.001, 0.05, 0.1, 0.2])
```


![png](images/Stock%20Prediction%20Project_21_0.png)



```
goog.changepoint_prior_validation(start_date='2016-01-04', end_date='2017-01-03', changepoint_priors=[0.001, 0.05, 0.1, 0.2])
```

    
    Validation Range 2016-01-04 00:00:00 to 2017-01-03 00:00:00.
    
         cps  train_err  train_range    test_err  test_range
    0  0.001  39.201088   119.332565   39.458360  119.175959
    1  0.050  12.057805    39.832086  204.848301  183.146491
    2  0.100  10.765865    36.778261  222.082001  285.733082
    3  0.200   9.441183    33.703218  211.435183  491.831858



![png](images/Stock%20Prediction%20Project_22_1.png)



![png](images/Stock%20Prediction%20Project_22_2.png)


## Evaluating Refined Model


```
goog.evaluate_prediction()
```

    
    Prediction Range: 2017-03-27 00:00:00 to 2018-03-27 00:00:00.
    
    Predicted price on 2018-03-24 00:00:00 = $1025.88.
    Actual price on    2018-03-23 00:00:00 = $1026.55.
    
    Average Absolute Error on Training Data = $10.92.
    Average Absolute Error on Testing  Data = $72.17.
    
    When the model predicted an increase, the price increased 58.74% of the time.
    When the model predicted a  decrease, the price decreased  49.06% of the time.
    
    The actual value was within the 80% confidence interval 80.00% of the time.



![png](images/Stock%20Prediction%20Project_24_1.png)


## Testing Our Luck in the Stock Market

Let's see how well our forecasts would play out in the real stock market.


```
goog.evaluate_prediction(nshares=1000)
```

    /content/Data-Analysis/stocker/stocker.py:613: SettingWithCopyWarning:
    
    
    A value is trying to be set on a copy of a slice from a DataFrame.
    Try using .loc[row_indexer,col_indexer] = value instead
    
    See the caveats in the documentation: http://pandas.pydata.org/pandas-docs/stable/indexing.html#indexing-view-versus-copy
    


    You played the stock market in GOOGL from 2017-03-27 00:00:00 to 2018-03-27 00:00:00 with 1000 shares.
    
    When the model predicted an increase, the price increased 58.74% of the time.
    When the model predicted a  decrease, the price decreased  49.06% of the time.
    
    The total profit using the Prophet model = $87380.00.
    The Buy and Hold strategy profit =         $188040.00.
    
    Thanks for playing the stock market!
    



![png](images/Stock%20Prediction%20Project_26_2.png)


This shows that it's better to simply invest for the long term.

## Preparing Data for More Machine Learning


```
# Getting the dataframe of the data

goog_data = goog.make_df('2004-08-19', '2018-03-27')
#2004-08-19 00:00:00 to 2018-03-27 00:00:00.
```


```
goog_data.head(50)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Date</th>
      <th>Open</th>
      <th>High</th>
      <th>Low</th>
      <th>Close</th>
      <th>Volume</th>
      <th>Ex-Dividend</th>
      <th>Split Ratio</th>
      <th>Adj. Open</th>
      <th>Adj. High</th>
      <th>Adj. Low</th>
      <th>Adj. Close</th>
      <th>Adj. Volume</th>
      <th>ds</th>
      <th>y</th>
      <th>Daily Change</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>2004-08-19</td>
      <td>100.010</td>
      <td>104.06</td>
      <td>95.96</td>
      <td>100.335</td>
      <td>44659000.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>50.159839</td>
      <td>52.191109</td>
      <td>48.128568</td>
      <td>50.322842</td>
      <td>44659000.0</td>
      <td>2004-08-19</td>
      <td>50.322842</td>
      <td>0.163003</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2004-08-20</td>
      <td>101.010</td>
      <td>109.08</td>
      <td>100.50</td>
      <td>108.310</td>
      <td>22834300.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>50.661387</td>
      <td>54.708881</td>
      <td>50.405597</td>
      <td>54.322689</td>
      <td>22834300.0</td>
      <td>2004-08-20</td>
      <td>54.322689</td>
      <td>3.661302</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2004-08-23</td>
      <td>110.760</td>
      <td>113.48</td>
      <td>109.05</td>
      <td>109.400</td>
      <td>18256100.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>55.551482</td>
      <td>56.915693</td>
      <td>54.693835</td>
      <td>54.869377</td>
      <td>18256100.0</td>
      <td>2004-08-23</td>
      <td>54.869377</td>
      <td>-0.682106</td>
    </tr>
    <tr>
      <th>3</th>
      <td>2004-08-24</td>
      <td>111.240</td>
      <td>111.60</td>
      <td>103.57</td>
      <td>104.870</td>
      <td>15247300.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>55.792225</td>
      <td>55.972783</td>
      <td>51.945350</td>
      <td>52.597363</td>
      <td>15247300.0</td>
      <td>2004-08-24</td>
      <td>52.597363</td>
      <td>-3.194862</td>
    </tr>
    <tr>
      <th>4</th>
      <td>2004-08-25</td>
      <td>104.760</td>
      <td>108.00</td>
      <td>103.88</td>
      <td>106.000</td>
      <td>9188600.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>52.542193</td>
      <td>54.167209</td>
      <td>52.100830</td>
      <td>53.164113</td>
      <td>9188600.0</td>
      <td>2004-08-25</td>
      <td>53.164113</td>
      <td>0.621920</td>
    </tr>
    <tr>
      <th>5</th>
      <td>2004-08-26</td>
      <td>104.950</td>
      <td>107.95</td>
      <td>104.66</td>
      <td>107.910</td>
      <td>7094800.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>52.637487</td>
      <td>54.142132</td>
      <td>52.492038</td>
      <td>54.122070</td>
      <td>7094800.0</td>
      <td>2004-08-26</td>
      <td>54.122070</td>
      <td>1.484583</td>
    </tr>
    <tr>
      <th>6</th>
      <td>2004-08-27</td>
      <td>108.100</td>
      <td>108.62</td>
      <td>105.69</td>
      <td>106.150</td>
      <td>6211700.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>54.217364</td>
      <td>54.478169</td>
      <td>53.008633</td>
      <td>53.239345</td>
      <td>6211700.0</td>
      <td>2004-08-27</td>
      <td>53.239345</td>
      <td>-0.978019</td>
    </tr>
    <tr>
      <th>7</th>
      <td>2004-08-30</td>
      <td>105.280</td>
      <td>105.49</td>
      <td>102.01</td>
      <td>102.010</td>
      <td>5196700.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>52.802998</td>
      <td>52.908323</td>
      <td>51.162935</td>
      <td>51.162935</td>
      <td>5196700.0</td>
      <td>2004-08-30</td>
      <td>51.162935</td>
      <td>-1.640063</td>
    </tr>
    <tr>
      <th>8</th>
      <td>2004-08-31</td>
      <td>102.320</td>
      <td>103.71</td>
      <td>102.16</td>
      <td>102.370</td>
      <td>4917800.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>51.318415</td>
      <td>52.015567</td>
      <td>51.238167</td>
      <td>51.343492</td>
      <td>4917800.0</td>
      <td>2004-08-31</td>
      <td>51.343492</td>
      <td>0.025077</td>
    </tr>
    <tr>
      <th>9</th>
      <td>2004-09-01</td>
      <td>102.700</td>
      <td>102.97</td>
      <td>99.67</td>
      <td>100.250</td>
      <td>9138200.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>51.509003</td>
      <td>51.644421</td>
      <td>49.989312</td>
      <td>50.280210</td>
      <td>9138200.0</td>
      <td>2004-09-01</td>
      <td>50.280210</td>
      <td>-1.228793</td>
    </tr>
    <tr>
      <th>10</th>
      <td>2004-09-02</td>
      <td>99.090</td>
      <td>102.37</td>
      <td>98.94</td>
      <td>101.510</td>
      <td>15118600.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>49.698414</td>
      <td>51.343492</td>
      <td>49.623182</td>
      <td>50.912161</td>
      <td>15118600.0</td>
      <td>2004-09-02</td>
      <td>50.912161</td>
      <td>1.213747</td>
    </tr>
    <tr>
      <th>11</th>
      <td>2004-09-03</td>
      <td>100.950</td>
      <td>101.74</td>
      <td>99.32</td>
      <td>100.010</td>
      <td>5152400.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>50.631294</td>
      <td>51.027517</td>
      <td>49.813770</td>
      <td>50.159839</td>
      <td>5152400.0</td>
      <td>2004-09-03</td>
      <td>50.159839</td>
      <td>-0.471455</td>
    </tr>
    <tr>
      <th>12</th>
      <td>2004-09-07</td>
      <td>101.010</td>
      <td>102.00</td>
      <td>99.61</td>
      <td>101.580</td>
      <td>5847500.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>50.661387</td>
      <td>51.157920</td>
      <td>49.959219</td>
      <td>50.947269</td>
      <td>5847500.0</td>
      <td>2004-09-07</td>
      <td>50.947269</td>
      <td>0.285882</td>
    </tr>
    <tr>
      <th>13</th>
      <td>2004-09-08</td>
      <td>100.740</td>
      <td>103.03</td>
      <td>100.50</td>
      <td>102.300</td>
      <td>4985600.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>50.525969</td>
      <td>51.674514</td>
      <td>50.405597</td>
      <td>51.308384</td>
      <td>4985600.0</td>
      <td>2004-09-08</td>
      <td>51.308384</td>
      <td>0.782415</td>
    </tr>
    <tr>
      <th>14</th>
      <td>2004-09-09</td>
      <td>102.500</td>
      <td>102.71</td>
      <td>101.00</td>
      <td>102.310</td>
      <td>4061700.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>51.408694</td>
      <td>51.514019</td>
      <td>50.656371</td>
      <td>51.313400</td>
      <td>4061700.0</td>
      <td>2004-09-09</td>
      <td>51.313400</td>
      <td>-0.095294</td>
    </tr>
    <tr>
      <th>15</th>
      <td>2004-09-10</td>
      <td>101.470</td>
      <td>106.56</td>
      <td>101.30</td>
      <td>105.330</td>
      <td>8698800.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>50.892099</td>
      <td>53.444980</td>
      <td>50.806836</td>
      <td>52.828075</td>
      <td>8698800.0</td>
      <td>2004-09-10</td>
      <td>52.828075</td>
      <td>1.935976</td>
    </tr>
    <tr>
      <th>16</th>
      <td>2004-09-13</td>
      <td>106.630</td>
      <td>108.41</td>
      <td>106.46</td>
      <td>107.500</td>
      <td>7844100.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>53.480088</td>
      <td>54.372844</td>
      <td>53.394825</td>
      <td>53.916435</td>
      <td>7844100.0</td>
      <td>2004-09-13</td>
      <td>53.916435</td>
      <td>0.436347</td>
    </tr>
    <tr>
      <th>17</th>
      <td>2004-09-14</td>
      <td>107.440</td>
      <td>112.00</td>
      <td>106.79</td>
      <td>111.490</td>
      <td>10828900.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>53.886342</td>
      <td>56.173402</td>
      <td>53.560336</td>
      <td>55.917612</td>
      <td>10828900.0</td>
      <td>2004-09-14</td>
      <td>55.917612</td>
      <td>2.031270</td>
    </tr>
    <tr>
      <th>18</th>
      <td>2004-09-15</td>
      <td>110.560</td>
      <td>114.23</td>
      <td>110.20</td>
      <td>112.000</td>
      <td>10713000.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>55.451172</td>
      <td>57.291854</td>
      <td>55.270615</td>
      <td>56.173402</td>
      <td>10713000.0</td>
      <td>2004-09-15</td>
      <td>56.173402</td>
      <td>0.722229</td>
    </tr>
    <tr>
      <th>19</th>
      <td>2004-09-16</td>
      <td>112.340</td>
      <td>115.80</td>
      <td>111.65</td>
      <td>113.970</td>
      <td>9266300.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>56.343928</td>
      <td>58.079285</td>
      <td>55.997860</td>
      <td>57.161452</td>
      <td>9266300.0</td>
      <td>2004-09-16</td>
      <td>57.161452</td>
      <td>0.817524</td>
    </tr>
    <tr>
      <th>20</th>
      <td>2004-09-17</td>
      <td>114.420</td>
      <td>117.49</td>
      <td>113.55</td>
      <td>117.490</td>
      <td>9472500.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>57.387149</td>
      <td>58.926902</td>
      <td>56.950802</td>
      <td>58.926902</td>
      <td>9472500.0</td>
      <td>2004-09-17</td>
      <td>58.926902</td>
      <td>1.539753</td>
    </tr>
    <tr>
      <th>21</th>
      <td>2004-09-20</td>
      <td>116.950</td>
      <td>121.60</td>
      <td>116.77</td>
      <td>119.360</td>
      <td>10628700.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>58.656066</td>
      <td>60.988265</td>
      <td>58.565787</td>
      <td>59.864797</td>
      <td>10628700.0</td>
      <td>2004-09-20</td>
      <td>59.864797</td>
      <td>1.208731</td>
    </tr>
    <tr>
      <th>22</th>
      <td>2004-09-21</td>
      <td>120.200</td>
      <td>120.42</td>
      <td>117.51</td>
      <td>117.840</td>
      <td>7228700.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>60.286097</td>
      <td>60.396438</td>
      <td>58.936933</td>
      <td>59.102444</td>
      <td>7228700.0</td>
      <td>2004-09-21</td>
      <td>59.102444</td>
      <td>-1.183654</td>
    </tr>
    <tr>
      <th>23</th>
      <td>2004-09-22</td>
      <td>117.450</td>
      <td>119.67</td>
      <td>116.81</td>
      <td>118.380</td>
      <td>7581200.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>58.906840</td>
      <td>60.020277</td>
      <td>58.585849</td>
      <td>59.373280</td>
      <td>7581200.0</td>
      <td>2004-09-22</td>
      <td>59.373280</td>
      <td>0.466440</td>
    </tr>
    <tr>
      <th>24</th>
      <td>2004-09-23</td>
      <td>118.840</td>
      <td>122.63</td>
      <td>117.02</td>
      <td>120.820</td>
      <td>8535600.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>59.603992</td>
      <td>61.504860</td>
      <td>58.691174</td>
      <td>60.597057</td>
      <td>8535600.0</td>
      <td>2004-09-23</td>
      <td>60.597057</td>
      <td>0.993065</td>
    </tr>
    <tr>
      <th>25</th>
      <td>2004-09-24</td>
      <td>120.970</td>
      <td>124.10</td>
      <td>119.76</td>
      <td>119.830</td>
      <td>9123400.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>60.672290</td>
      <td>62.242136</td>
      <td>60.065416</td>
      <td>60.100525</td>
      <td>9123400.0</td>
      <td>2004-09-24</td>
      <td>60.100525</td>
      <td>-0.571765</td>
    </tr>
    <tr>
      <th>26</th>
      <td>2004-09-27</td>
      <td>119.560</td>
      <td>120.88</td>
      <td>117.80</td>
      <td>118.260</td>
      <td>7066100.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>59.965107</td>
      <td>60.627150</td>
      <td>59.082382</td>
      <td>59.313094</td>
      <td>7066100.0</td>
      <td>2004-09-27</td>
      <td>59.313094</td>
      <td>-0.652013</td>
    </tr>
    <tr>
      <th>27</th>
      <td>2004-09-28</td>
      <td>121.150</td>
      <td>127.40</td>
      <td>120.21</td>
      <td>126.860</td>
      <td>16929000.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>60.762568</td>
      <td>63.897245</td>
      <td>60.291113</td>
      <td>63.626409</td>
      <td>16929000.0</td>
      <td>2004-09-28</td>
      <td>63.626409</td>
      <td>2.863840</td>
    </tr>
    <tr>
      <th>28</th>
      <td>2004-09-29</td>
      <td>126.530</td>
      <td>135.02</td>
      <td>126.23</td>
      <td>131.080</td>
      <td>30516400.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>63.460898</td>
      <td>67.719042</td>
      <td>63.310433</td>
      <td>65.742942</td>
      <td>30516400.0</td>
      <td>2004-09-29</td>
      <td>65.742942</td>
      <td>2.282044</td>
    </tr>
    <tr>
      <th>29</th>
      <td>2004-09-30</td>
      <td>129.899</td>
      <td>132.30</td>
      <td>129.00</td>
      <td>129.600</td>
      <td>13758000.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>65.150614</td>
      <td>66.354831</td>
      <td>64.699722</td>
      <td>65.000651</td>
      <td>13758000.0</td>
      <td>2004-09-30</td>
      <td>65.000651</td>
      <td>-0.149963</td>
    </tr>
    <tr>
      <th>30</th>
      <td>2004-10-01</td>
      <td>130.800</td>
      <td>134.24</td>
      <td>128.90</td>
      <td>132.580</td>
      <td>15124800.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>65.602509</td>
      <td>67.327835</td>
      <td>64.649567</td>
      <td>66.495265</td>
      <td>15124800.0</td>
      <td>2004-10-01</td>
      <td>66.495265</td>
      <td>0.892756</td>
    </tr>
    <tr>
      <th>31</th>
      <td>2004-10-04</td>
      <td>135.275</td>
      <td>136.87</td>
      <td>134.03</td>
      <td>135.060</td>
      <td>13022700.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>67.846937</td>
      <td>68.646906</td>
      <td>67.222509</td>
      <td>67.739104</td>
      <td>13022700.0</td>
      <td>2004-10-04</td>
      <td>67.739104</td>
      <td>-0.107833</td>
    </tr>
    <tr>
      <th>32</th>
      <td>2004-10-05</td>
      <td>134.660</td>
      <td>138.53</td>
      <td>132.24</td>
      <td>138.370</td>
      <td>14973200.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>67.538485</td>
      <td>69.479476</td>
      <td>66.324738</td>
      <td>69.399229</td>
      <td>14973200.0</td>
      <td>2004-10-05</td>
      <td>69.399229</td>
      <td>1.860744</td>
    </tr>
    <tr>
      <th>33</th>
      <td>2004-10-06</td>
      <td>137.670</td>
      <td>138.45</td>
      <td>136.00</td>
      <td>137.080</td>
      <td>13381400.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>69.048145</td>
      <td>69.439353</td>
      <td>68.210559</td>
      <td>68.752232</td>
      <td>13381400.0</td>
      <td>2004-10-06</td>
      <td>68.752232</td>
      <td>-0.295913</td>
    </tr>
    <tr>
      <th>34</th>
      <td>2004-10-07</td>
      <td>136.560</td>
      <td>139.88</td>
      <td>136.55</td>
      <td>138.850</td>
      <td>14115000.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>68.491426</td>
      <td>70.156567</td>
      <td>68.486411</td>
      <td>69.639972</td>
      <td>14115000.0</td>
      <td>2004-10-07</td>
      <td>69.639972</td>
      <td>1.148545</td>
    </tr>
    <tr>
      <th>35</th>
      <td>2004-10-08</td>
      <td>138.730</td>
      <td>139.68</td>
      <td>137.02</td>
      <td>137.730</td>
      <td>11069500.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>69.579786</td>
      <td>70.056257</td>
      <td>68.722139</td>
      <td>69.078238</td>
      <td>11069500.0</td>
      <td>2004-10-08</td>
      <td>69.078238</td>
      <td>-0.501548</td>
    </tr>
    <tr>
      <th>36</th>
      <td>2004-10-11</td>
      <td>137.010</td>
      <td>138.86</td>
      <td>133.85</td>
      <td>135.260</td>
      <td>10472100.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>68.717123</td>
      <td>69.644987</td>
      <td>67.132231</td>
      <td>67.839414</td>
      <td>10472100.0</td>
      <td>2004-10-11</td>
      <td>67.839414</td>
      <td>-0.877709</td>
    </tr>
    <tr>
      <th>37</th>
      <td>2004-10-12</td>
      <td>134.490</td>
      <td>137.61</td>
      <td>133.40</td>
      <td>137.400</td>
      <td>11665500.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>67.453222</td>
      <td>69.018052</td>
      <td>66.906534</td>
      <td>68.912727</td>
      <td>11665500.0</td>
      <td>2004-10-12</td>
      <td>68.912727</td>
      <td>1.459505</td>
    </tr>
    <tr>
      <th>38</th>
      <td>2004-10-13</td>
      <td>143.240</td>
      <td>143.55</td>
      <td>140.08</td>
      <td>140.900</td>
      <td>19766200.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>71.841769</td>
      <td>71.997249</td>
      <td>70.256876</td>
      <td>70.668146</td>
      <td>19766200.0</td>
      <td>2004-10-13</td>
      <td>70.668146</td>
      <td>-1.173623</td>
    </tr>
    <tr>
      <th>39</th>
      <td>2004-10-14</td>
      <td>141.020</td>
      <td>142.38</td>
      <td>138.56</td>
      <td>142.000</td>
      <td>10442100.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>70.728332</td>
      <td>71.410437</td>
      <td>69.494523</td>
      <td>71.219849</td>
      <td>10442100.0</td>
      <td>2004-10-14</td>
      <td>71.219849</td>
      <td>0.491517</td>
    </tr>
    <tr>
      <th>40</th>
      <td>2004-10-15</td>
      <td>144.950</td>
      <td>145.50</td>
      <td>141.95</td>
      <td>144.110</td>
      <td>13194700.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>72.699416</td>
      <td>72.975268</td>
      <td>71.194771</td>
      <td>72.278116</td>
      <td>13194700.0</td>
      <td>2004-10-15</td>
      <td>72.278116</td>
      <td>-0.421301</td>
    </tr>
    <tr>
      <th>41</th>
      <td>2004-10-18</td>
      <td>143.120</td>
      <td>149.20</td>
      <td>141.21</td>
      <td>149.160</td>
      <td>14036300.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>71.781583</td>
      <td>74.830996</td>
      <td>70.823626</td>
      <td>74.810934</td>
      <td>14036300.0</td>
      <td>2004-10-18</td>
      <td>74.810934</td>
      <td>3.029351</td>
    </tr>
    <tr>
      <th>42</th>
      <td>2004-10-19</td>
      <td>150.500</td>
      <td>152.40</td>
      <td>147.35</td>
      <td>147.940</td>
      <td>18109800.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>75.483009</td>
      <td>76.435950</td>
      <td>73.903132</td>
      <td>74.199045</td>
      <td>18109800.0</td>
      <td>2004-10-19</td>
      <td>74.199045</td>
      <td>-1.283963</td>
    </tr>
    <tr>
      <th>43</th>
      <td>2004-10-20</td>
      <td>147.940</td>
      <td>148.99</td>
      <td>139.60</td>
      <td>140.490</td>
      <td>22722600.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>74.199045</td>
      <td>74.725671</td>
      <td>70.016133</td>
      <td>70.462511</td>
      <td>22722600.0</td>
      <td>2004-10-20</td>
      <td>70.462511</td>
      <td>-3.736534</td>
    </tr>
    <tr>
      <th>44</th>
      <td>2004-10-21</td>
      <td>144.130</td>
      <td>150.13</td>
      <td>141.62</td>
      <td>149.380</td>
      <td>29149800.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>72.288147</td>
      <td>75.297436</td>
      <td>71.029261</td>
      <td>74.921275</td>
      <td>29149800.0</td>
      <td>2004-10-21</td>
      <td>74.921275</td>
      <td>2.633128</td>
    </tr>
    <tr>
      <th>45</th>
      <td>2004-10-22</td>
      <td>170.435</td>
      <td>180.17</td>
      <td>164.08</td>
      <td>172.430</td>
      <td>73710000.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>85.481373</td>
      <td>90.363945</td>
      <td>82.294034</td>
      <td>86.481962</td>
      <td>73710000.0</td>
      <td>2004-10-22</td>
      <td>86.481962</td>
      <td>1.000589</td>
    </tr>
    <tr>
      <th>46</th>
      <td>2004-10-25</td>
      <td>176.280</td>
      <td>194.43</td>
      <td>172.55</td>
      <td>187.400</td>
      <td>65462800.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>88.412922</td>
      <td>97.516023</td>
      <td>86.542147</td>
      <td>93.990139</td>
      <td>65462800.0</td>
      <td>2004-10-25</td>
      <td>93.990139</td>
      <td>5.577216</td>
    </tr>
    <tr>
      <th>47</th>
      <td>2004-10-26</td>
      <td>186.449</td>
      <td>192.64</td>
      <td>180.00</td>
      <td>181.800</td>
      <td>44569500.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>93.513166</td>
      <td>96.618251</td>
      <td>90.278682</td>
      <td>91.181468</td>
      <td>44569500.0</td>
      <td>2004-10-26</td>
      <td>91.181468</td>
      <td>-2.331698</td>
    </tr>
    <tr>
      <th>48</th>
      <td>2004-10-27</td>
      <td>182.509</td>
      <td>189.52</td>
      <td>181.77</td>
      <td>185.970</td>
      <td>26686200.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>91.537066</td>
      <td>95.053421</td>
      <td>91.166422</td>
      <td>93.272925</td>
      <td>26686200.0</td>
      <td>2004-10-27</td>
      <td>93.272925</td>
      <td>1.735858</td>
    </tr>
    <tr>
      <th>49</th>
      <td>2004-10-28</td>
      <td>186.630</td>
      <td>194.39</td>
      <td>185.60</td>
      <td>193.300</td>
      <td>29663900.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>93.603946</td>
      <td>97.495961</td>
      <td>93.087352</td>
      <td>96.949273</td>
      <td>29663900.0</td>
      <td>2004-10-28</td>
      <td>96.949273</td>
      <td>3.345327</td>
    </tr>
  </tbody>
</table>
</div>




```
goog_data = goog_data[['Date', 'Open', 'High', 'Low', 'Close', 'Adj. Close', 'Volume']]
```


```
goog_data.head(50)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Date</th>
      <th>Open</th>
      <th>High</th>
      <th>Low</th>
      <th>Close</th>
      <th>Adj. Close</th>
      <th>Volume</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>2004-08-19</td>
      <td>100.010</td>
      <td>104.06</td>
      <td>95.96</td>
      <td>100.335</td>
      <td>50.322842</td>
      <td>44659000.0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2004-08-20</td>
      <td>101.010</td>
      <td>109.08</td>
      <td>100.50</td>
      <td>108.310</td>
      <td>54.322689</td>
      <td>22834300.0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2004-08-23</td>
      <td>110.760</td>
      <td>113.48</td>
      <td>109.05</td>
      <td>109.400</td>
      <td>54.869377</td>
      <td>18256100.0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>2004-08-24</td>
      <td>111.240</td>
      <td>111.60</td>
      <td>103.57</td>
      <td>104.870</td>
      <td>52.597363</td>
      <td>15247300.0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>2004-08-25</td>
      <td>104.760</td>
      <td>108.00</td>
      <td>103.88</td>
      <td>106.000</td>
      <td>53.164113</td>
      <td>9188600.0</td>
    </tr>
    <tr>
      <th>5</th>
      <td>2004-08-26</td>
      <td>104.950</td>
      <td>107.95</td>
      <td>104.66</td>
      <td>107.910</td>
      <td>54.122070</td>
      <td>7094800.0</td>
    </tr>
    <tr>
      <th>6</th>
      <td>2004-08-27</td>
      <td>108.100</td>
      <td>108.62</td>
      <td>105.69</td>
      <td>106.150</td>
      <td>53.239345</td>
      <td>6211700.0</td>
    </tr>
    <tr>
      <th>7</th>
      <td>2004-08-30</td>
      <td>105.280</td>
      <td>105.49</td>
      <td>102.01</td>
      <td>102.010</td>
      <td>51.162935</td>
      <td>5196700.0</td>
    </tr>
    <tr>
      <th>8</th>
      <td>2004-08-31</td>
      <td>102.320</td>
      <td>103.71</td>
      <td>102.16</td>
      <td>102.370</td>
      <td>51.343492</td>
      <td>4917800.0</td>
    </tr>
    <tr>
      <th>9</th>
      <td>2004-09-01</td>
      <td>102.700</td>
      <td>102.97</td>
      <td>99.67</td>
      <td>100.250</td>
      <td>50.280210</td>
      <td>9138200.0</td>
    </tr>
    <tr>
      <th>10</th>
      <td>2004-09-02</td>
      <td>99.090</td>
      <td>102.37</td>
      <td>98.94</td>
      <td>101.510</td>
      <td>50.912161</td>
      <td>15118600.0</td>
    </tr>
    <tr>
      <th>11</th>
      <td>2004-09-03</td>
      <td>100.950</td>
      <td>101.74</td>
      <td>99.32</td>
      <td>100.010</td>
      <td>50.159839</td>
      <td>5152400.0</td>
    </tr>
    <tr>
      <th>12</th>
      <td>2004-09-07</td>
      <td>101.010</td>
      <td>102.00</td>
      <td>99.61</td>
      <td>101.580</td>
      <td>50.947269</td>
      <td>5847500.0</td>
    </tr>
    <tr>
      <th>13</th>
      <td>2004-09-08</td>
      <td>100.740</td>
      <td>103.03</td>
      <td>100.50</td>
      <td>102.300</td>
      <td>51.308384</td>
      <td>4985600.0</td>
    </tr>
    <tr>
      <th>14</th>
      <td>2004-09-09</td>
      <td>102.500</td>
      <td>102.71</td>
      <td>101.00</td>
      <td>102.310</td>
      <td>51.313400</td>
      <td>4061700.0</td>
    </tr>
    <tr>
      <th>15</th>
      <td>2004-09-10</td>
      <td>101.470</td>
      <td>106.56</td>
      <td>101.30</td>
      <td>105.330</td>
      <td>52.828075</td>
      <td>8698800.0</td>
    </tr>
    <tr>
      <th>16</th>
      <td>2004-09-13</td>
      <td>106.630</td>
      <td>108.41</td>
      <td>106.46</td>
      <td>107.500</td>
      <td>53.916435</td>
      <td>7844100.0</td>
    </tr>
    <tr>
      <th>17</th>
      <td>2004-09-14</td>
      <td>107.440</td>
      <td>112.00</td>
      <td>106.79</td>
      <td>111.490</td>
      <td>55.917612</td>
      <td>10828900.0</td>
    </tr>
    <tr>
      <th>18</th>
      <td>2004-09-15</td>
      <td>110.560</td>
      <td>114.23</td>
      <td>110.20</td>
      <td>112.000</td>
      <td>56.173402</td>
      <td>10713000.0</td>
    </tr>
    <tr>
      <th>19</th>
      <td>2004-09-16</td>
      <td>112.340</td>
      <td>115.80</td>
      <td>111.65</td>
      <td>113.970</td>
      <td>57.161452</td>
      <td>9266300.0</td>
    </tr>
    <tr>
      <th>20</th>
      <td>2004-09-17</td>
      <td>114.420</td>
      <td>117.49</td>
      <td>113.55</td>
      <td>117.490</td>
      <td>58.926902</td>
      <td>9472500.0</td>
    </tr>
    <tr>
      <th>21</th>
      <td>2004-09-20</td>
      <td>116.950</td>
      <td>121.60</td>
      <td>116.77</td>
      <td>119.360</td>
      <td>59.864797</td>
      <td>10628700.0</td>
    </tr>
    <tr>
      <th>22</th>
      <td>2004-09-21</td>
      <td>120.200</td>
      <td>120.42</td>
      <td>117.51</td>
      <td>117.840</td>
      <td>59.102444</td>
      <td>7228700.0</td>
    </tr>
    <tr>
      <th>23</th>
      <td>2004-09-22</td>
      <td>117.450</td>
      <td>119.67</td>
      <td>116.81</td>
      <td>118.380</td>
      <td>59.373280</td>
      <td>7581200.0</td>
    </tr>
    <tr>
      <th>24</th>
      <td>2004-09-23</td>
      <td>118.840</td>
      <td>122.63</td>
      <td>117.02</td>
      <td>120.820</td>
      <td>60.597057</td>
      <td>8535600.0</td>
    </tr>
    <tr>
      <th>25</th>
      <td>2004-09-24</td>
      <td>120.970</td>
      <td>124.10</td>
      <td>119.76</td>
      <td>119.830</td>
      <td>60.100525</td>
      <td>9123400.0</td>
    </tr>
    <tr>
      <th>26</th>
      <td>2004-09-27</td>
      <td>119.560</td>
      <td>120.88</td>
      <td>117.80</td>
      <td>118.260</td>
      <td>59.313094</td>
      <td>7066100.0</td>
    </tr>
    <tr>
      <th>27</th>
      <td>2004-09-28</td>
      <td>121.150</td>
      <td>127.40</td>
      <td>120.21</td>
      <td>126.860</td>
      <td>63.626409</td>
      <td>16929000.0</td>
    </tr>
    <tr>
      <th>28</th>
      <td>2004-09-29</td>
      <td>126.530</td>
      <td>135.02</td>
      <td>126.23</td>
      <td>131.080</td>
      <td>65.742942</td>
      <td>30516400.0</td>
    </tr>
    <tr>
      <th>29</th>
      <td>2004-09-30</td>
      <td>129.899</td>
      <td>132.30</td>
      <td>129.00</td>
      <td>129.600</td>
      <td>65.000651</td>
      <td>13758000.0</td>
    </tr>
    <tr>
      <th>30</th>
      <td>2004-10-01</td>
      <td>130.800</td>
      <td>134.24</td>
      <td>128.90</td>
      <td>132.580</td>
      <td>66.495265</td>
      <td>15124800.0</td>
    </tr>
    <tr>
      <th>31</th>
      <td>2004-10-04</td>
      <td>135.275</td>
      <td>136.87</td>
      <td>134.03</td>
      <td>135.060</td>
      <td>67.739104</td>
      <td>13022700.0</td>
    </tr>
    <tr>
      <th>32</th>
      <td>2004-10-05</td>
      <td>134.660</td>
      <td>138.53</td>
      <td>132.24</td>
      <td>138.370</td>
      <td>69.399229</td>
      <td>14973200.0</td>
    </tr>
    <tr>
      <th>33</th>
      <td>2004-10-06</td>
      <td>137.670</td>
      <td>138.45</td>
      <td>136.00</td>
      <td>137.080</td>
      <td>68.752232</td>
      <td>13381400.0</td>
    </tr>
    <tr>
      <th>34</th>
      <td>2004-10-07</td>
      <td>136.560</td>
      <td>139.88</td>
      <td>136.55</td>
      <td>138.850</td>
      <td>69.639972</td>
      <td>14115000.0</td>
    </tr>
    <tr>
      <th>35</th>
      <td>2004-10-08</td>
      <td>138.730</td>
      <td>139.68</td>
      <td>137.02</td>
      <td>137.730</td>
      <td>69.078238</td>
      <td>11069500.0</td>
    </tr>
    <tr>
      <th>36</th>
      <td>2004-10-11</td>
      <td>137.010</td>
      <td>138.86</td>
      <td>133.85</td>
      <td>135.260</td>
      <td>67.839414</td>
      <td>10472100.0</td>
    </tr>
    <tr>
      <th>37</th>
      <td>2004-10-12</td>
      <td>134.490</td>
      <td>137.61</td>
      <td>133.40</td>
      <td>137.400</td>
      <td>68.912727</td>
      <td>11665500.0</td>
    </tr>
    <tr>
      <th>38</th>
      <td>2004-10-13</td>
      <td>143.240</td>
      <td>143.55</td>
      <td>140.08</td>
      <td>140.900</td>
      <td>70.668146</td>
      <td>19766200.0</td>
    </tr>
    <tr>
      <th>39</th>
      <td>2004-10-14</td>
      <td>141.020</td>
      <td>142.38</td>
      <td>138.56</td>
      <td>142.000</td>
      <td>71.219849</td>
      <td>10442100.0</td>
    </tr>
    <tr>
      <th>40</th>
      <td>2004-10-15</td>
      <td>144.950</td>
      <td>145.50</td>
      <td>141.95</td>
      <td>144.110</td>
      <td>72.278116</td>
      <td>13194700.0</td>
    </tr>
    <tr>
      <th>41</th>
      <td>2004-10-18</td>
      <td>143.120</td>
      <td>149.20</td>
      <td>141.21</td>
      <td>149.160</td>
      <td>74.810934</td>
      <td>14036300.0</td>
    </tr>
    <tr>
      <th>42</th>
      <td>2004-10-19</td>
      <td>150.500</td>
      <td>152.40</td>
      <td>147.35</td>
      <td>147.940</td>
      <td>74.199045</td>
      <td>18109800.0</td>
    </tr>
    <tr>
      <th>43</th>
      <td>2004-10-20</td>
      <td>147.940</td>
      <td>148.99</td>
      <td>139.60</td>
      <td>140.490</td>
      <td>70.462511</td>
      <td>22722600.0</td>
    </tr>
    <tr>
      <th>44</th>
      <td>2004-10-21</td>
      <td>144.130</td>
      <td>150.13</td>
      <td>141.62</td>
      <td>149.380</td>
      <td>74.921275</td>
      <td>29149800.0</td>
    </tr>
    <tr>
      <th>45</th>
      <td>2004-10-22</td>
      <td>170.435</td>
      <td>180.17</td>
      <td>164.08</td>
      <td>172.430</td>
      <td>86.481962</td>
      <td>73710000.0</td>
    </tr>
    <tr>
      <th>46</th>
      <td>2004-10-25</td>
      <td>176.280</td>
      <td>194.43</td>
      <td>172.55</td>
      <td>187.400</td>
      <td>93.990139</td>
      <td>65462800.0</td>
    </tr>
    <tr>
      <th>47</th>
      <td>2004-10-26</td>
      <td>186.449</td>
      <td>192.64</td>
      <td>180.00</td>
      <td>181.800</td>
      <td>91.181468</td>
      <td>44569500.0</td>
    </tr>
    <tr>
      <th>48</th>
      <td>2004-10-27</td>
      <td>182.509</td>
      <td>189.52</td>
      <td>181.77</td>
      <td>185.970</td>
      <td>93.272925</td>
      <td>26686200.0</td>
    </tr>
    <tr>
      <th>49</th>
      <td>2004-10-28</td>
      <td>186.630</td>
      <td>194.39</td>
      <td>185.60</td>
      <td>193.300</td>
      <td>96.949273</td>
      <td>29663900.0</td>
    </tr>
  </tbody>
</table>
</div>



# Moving Average


```
import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
%matplotlib inline
```


```
import matplotlib.style
import matplotlib as mpl
mpl.style.use('ggplot')
```


```
from matplotlib.pylab import rcParams
rcParams['figure.figsize'] = 20, 10
```


```
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler(feature_range=(0, 1))


```


```
# Creating copy of goog_data dataframe for moving averages

df = goog_data
```


```
df.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Date</th>
      <th>Open</th>
      <th>High</th>
      <th>Low</th>
      <th>Close</th>
      <th>Adj. Close</th>
      <th>Volume</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>2004-08-19</td>
      <td>100.01</td>
      <td>104.06</td>
      <td>95.96</td>
      <td>100.335</td>
      <td>50.322842</td>
      <td>44659000.0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2004-08-20</td>
      <td>101.01</td>
      <td>109.08</td>
      <td>100.50</td>
      <td>108.310</td>
      <td>54.322689</td>
      <td>22834300.0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2004-08-23</td>
      <td>110.76</td>
      <td>113.48</td>
      <td>109.05</td>
      <td>109.400</td>
      <td>54.869377</td>
      <td>18256100.0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>2004-08-24</td>
      <td>111.24</td>
      <td>111.60</td>
      <td>103.57</td>
      <td>104.870</td>
      <td>52.597363</td>
      <td>15247300.0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>2004-08-25</td>
      <td>104.76</td>
      <td>108.00</td>
      <td>103.88</td>
      <td>106.000</td>
      <td>53.164113</td>
      <td>9188600.0</td>
    </tr>
  </tbody>
</table>
</div>




```
df['Date'] = pd.to_datetime(df.Date, format='%Y-%m-%d')
df.index = df['Date']
```


```
df.head(50)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Date</th>
      <th>Open</th>
      <th>High</th>
      <th>Low</th>
      <th>Close</th>
      <th>Adj. Close</th>
      <th>Volume</th>
    </tr>
    <tr>
      <th>Date</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>2004-08-19</th>
      <td>2004-08-19</td>
      <td>100.010</td>
      <td>104.06</td>
      <td>95.96</td>
      <td>100.335</td>
      <td>50.322842</td>
      <td>44659000.0</td>
    </tr>
    <tr>
      <th>2004-08-20</th>
      <td>2004-08-20</td>
      <td>101.010</td>
      <td>109.08</td>
      <td>100.50</td>
      <td>108.310</td>
      <td>54.322689</td>
      <td>22834300.0</td>
    </tr>
    <tr>
      <th>2004-08-23</th>
      <td>2004-08-23</td>
      <td>110.760</td>
      <td>113.48</td>
      <td>109.05</td>
      <td>109.400</td>
      <td>54.869377</td>
      <td>18256100.0</td>
    </tr>
    <tr>
      <th>2004-08-24</th>
      <td>2004-08-24</td>
      <td>111.240</td>
      <td>111.60</td>
      <td>103.57</td>
      <td>104.870</td>
      <td>52.597363</td>
      <td>15247300.0</td>
    </tr>
    <tr>
      <th>2004-08-25</th>
      <td>2004-08-25</td>
      <td>104.760</td>
      <td>108.00</td>
      <td>103.88</td>
      <td>106.000</td>
      <td>53.164113</td>
      <td>9188600.0</td>
    </tr>
    <tr>
      <th>2004-08-26</th>
      <td>2004-08-26</td>
      <td>104.950</td>
      <td>107.95</td>
      <td>104.66</td>
      <td>107.910</td>
      <td>54.122070</td>
      <td>7094800.0</td>
    </tr>
    <tr>
      <th>2004-08-27</th>
      <td>2004-08-27</td>
      <td>108.100</td>
      <td>108.62</td>
      <td>105.69</td>
      <td>106.150</td>
      <td>53.239345</td>
      <td>6211700.0</td>
    </tr>
    <tr>
      <th>2004-08-30</th>
      <td>2004-08-30</td>
      <td>105.280</td>
      <td>105.49</td>
      <td>102.01</td>
      <td>102.010</td>
      <td>51.162935</td>
      <td>5196700.0</td>
    </tr>
    <tr>
      <th>2004-08-31</th>
      <td>2004-08-31</td>
      <td>102.320</td>
      <td>103.71</td>
      <td>102.16</td>
      <td>102.370</td>
      <td>51.343492</td>
      <td>4917800.0</td>
    </tr>
    <tr>
      <th>2004-09-01</th>
      <td>2004-09-01</td>
      <td>102.700</td>
      <td>102.97</td>
      <td>99.67</td>
      <td>100.250</td>
      <td>50.280210</td>
      <td>9138200.0</td>
    </tr>
    <tr>
      <th>2004-09-02</th>
      <td>2004-09-02</td>
      <td>99.090</td>
      <td>102.37</td>
      <td>98.94</td>
      <td>101.510</td>
      <td>50.912161</td>
      <td>15118600.0</td>
    </tr>
    <tr>
      <th>2004-09-03</th>
      <td>2004-09-03</td>
      <td>100.950</td>
      <td>101.74</td>
      <td>99.32</td>
      <td>100.010</td>
      <td>50.159839</td>
      <td>5152400.0</td>
    </tr>
    <tr>
      <th>2004-09-07</th>
      <td>2004-09-07</td>
      <td>101.010</td>
      <td>102.00</td>
      <td>99.61</td>
      <td>101.580</td>
      <td>50.947269</td>
      <td>5847500.0</td>
    </tr>
    <tr>
      <th>2004-09-08</th>
      <td>2004-09-08</td>
      <td>100.740</td>
      <td>103.03</td>
      <td>100.50</td>
      <td>102.300</td>
      <td>51.308384</td>
      <td>4985600.0</td>
    </tr>
    <tr>
      <th>2004-09-09</th>
      <td>2004-09-09</td>
      <td>102.500</td>
      <td>102.71</td>
      <td>101.00</td>
      <td>102.310</td>
      <td>51.313400</td>
      <td>4061700.0</td>
    </tr>
    <tr>
      <th>2004-09-10</th>
      <td>2004-09-10</td>
      <td>101.470</td>
      <td>106.56</td>
      <td>101.30</td>
      <td>105.330</td>
      <td>52.828075</td>
      <td>8698800.0</td>
    </tr>
    <tr>
      <th>2004-09-13</th>
      <td>2004-09-13</td>
      <td>106.630</td>
      <td>108.41</td>
      <td>106.46</td>
      <td>107.500</td>
      <td>53.916435</td>
      <td>7844100.0</td>
    </tr>
    <tr>
      <th>2004-09-14</th>
      <td>2004-09-14</td>
      <td>107.440</td>
      <td>112.00</td>
      <td>106.79</td>
      <td>111.490</td>
      <td>55.917612</td>
      <td>10828900.0</td>
    </tr>
    <tr>
      <th>2004-09-15</th>
      <td>2004-09-15</td>
      <td>110.560</td>
      <td>114.23</td>
      <td>110.20</td>
      <td>112.000</td>
      <td>56.173402</td>
      <td>10713000.0</td>
    </tr>
    <tr>
      <th>2004-09-16</th>
      <td>2004-09-16</td>
      <td>112.340</td>
      <td>115.80</td>
      <td>111.65</td>
      <td>113.970</td>
      <td>57.161452</td>
      <td>9266300.0</td>
    </tr>
    <tr>
      <th>2004-09-17</th>
      <td>2004-09-17</td>
      <td>114.420</td>
      <td>117.49</td>
      <td>113.55</td>
      <td>117.490</td>
      <td>58.926902</td>
      <td>9472500.0</td>
    </tr>
    <tr>
      <th>2004-09-20</th>
      <td>2004-09-20</td>
      <td>116.950</td>
      <td>121.60</td>
      <td>116.77</td>
      <td>119.360</td>
      <td>59.864797</td>
      <td>10628700.0</td>
    </tr>
    <tr>
      <th>2004-09-21</th>
      <td>2004-09-21</td>
      <td>120.200</td>
      <td>120.42</td>
      <td>117.51</td>
      <td>117.840</td>
      <td>59.102444</td>
      <td>7228700.0</td>
    </tr>
    <tr>
      <th>2004-09-22</th>
      <td>2004-09-22</td>
      <td>117.450</td>
      <td>119.67</td>
      <td>116.81</td>
      <td>118.380</td>
      <td>59.373280</td>
      <td>7581200.0</td>
    </tr>
    <tr>
      <th>2004-09-23</th>
      <td>2004-09-23</td>
      <td>118.840</td>
      <td>122.63</td>
      <td>117.02</td>
      <td>120.820</td>
      <td>60.597057</td>
      <td>8535600.0</td>
    </tr>
    <tr>
      <th>2004-09-24</th>
      <td>2004-09-24</td>
      <td>120.970</td>
      <td>124.10</td>
      <td>119.76</td>
      <td>119.830</td>
      <td>60.100525</td>
      <td>9123400.0</td>
    </tr>
    <tr>
      <th>2004-09-27</th>
      <td>2004-09-27</td>
      <td>119.560</td>
      <td>120.88</td>
      <td>117.80</td>
      <td>118.260</td>
      <td>59.313094</td>
      <td>7066100.0</td>
    </tr>
    <tr>
      <th>2004-09-28</th>
      <td>2004-09-28</td>
      <td>121.150</td>
      <td>127.40</td>
      <td>120.21</td>
      <td>126.860</td>
      <td>63.626409</td>
      <td>16929000.0</td>
    </tr>
    <tr>
      <th>2004-09-29</th>
      <td>2004-09-29</td>
      <td>126.530</td>
      <td>135.02</td>
      <td>126.23</td>
      <td>131.080</td>
      <td>65.742942</td>
      <td>30516400.0</td>
    </tr>
    <tr>
      <th>2004-09-30</th>
      <td>2004-09-30</td>
      <td>129.899</td>
      <td>132.30</td>
      <td>129.00</td>
      <td>129.600</td>
      <td>65.000651</td>
      <td>13758000.0</td>
    </tr>
    <tr>
      <th>2004-10-01</th>
      <td>2004-10-01</td>
      <td>130.800</td>
      <td>134.24</td>
      <td>128.90</td>
      <td>132.580</td>
      <td>66.495265</td>
      <td>15124800.0</td>
    </tr>
    <tr>
      <th>2004-10-04</th>
      <td>2004-10-04</td>
      <td>135.275</td>
      <td>136.87</td>
      <td>134.03</td>
      <td>135.060</td>
      <td>67.739104</td>
      <td>13022700.0</td>
    </tr>
    <tr>
      <th>2004-10-05</th>
      <td>2004-10-05</td>
      <td>134.660</td>
      <td>138.53</td>
      <td>132.24</td>
      <td>138.370</td>
      <td>69.399229</td>
      <td>14973200.0</td>
    </tr>
    <tr>
      <th>2004-10-06</th>
      <td>2004-10-06</td>
      <td>137.670</td>
      <td>138.45</td>
      <td>136.00</td>
      <td>137.080</td>
      <td>68.752232</td>
      <td>13381400.0</td>
    </tr>
    <tr>
      <th>2004-10-07</th>
      <td>2004-10-07</td>
      <td>136.560</td>
      <td>139.88</td>
      <td>136.55</td>
      <td>138.850</td>
      <td>69.639972</td>
      <td>14115000.0</td>
    </tr>
    <tr>
      <th>2004-10-08</th>
      <td>2004-10-08</td>
      <td>138.730</td>
      <td>139.68</td>
      <td>137.02</td>
      <td>137.730</td>
      <td>69.078238</td>
      <td>11069500.0</td>
    </tr>
    <tr>
      <th>2004-10-11</th>
      <td>2004-10-11</td>
      <td>137.010</td>
      <td>138.86</td>
      <td>133.85</td>
      <td>135.260</td>
      <td>67.839414</td>
      <td>10472100.0</td>
    </tr>
    <tr>
      <th>2004-10-12</th>
      <td>2004-10-12</td>
      <td>134.490</td>
      <td>137.61</td>
      <td>133.40</td>
      <td>137.400</td>
      <td>68.912727</td>
      <td>11665500.0</td>
    </tr>
    <tr>
      <th>2004-10-13</th>
      <td>2004-10-13</td>
      <td>143.240</td>
      <td>143.55</td>
      <td>140.08</td>
      <td>140.900</td>
      <td>70.668146</td>
      <td>19766200.0</td>
    </tr>
    <tr>
      <th>2004-10-14</th>
      <td>2004-10-14</td>
      <td>141.020</td>
      <td>142.38</td>
      <td>138.56</td>
      <td>142.000</td>
      <td>71.219849</td>
      <td>10442100.0</td>
    </tr>
    <tr>
      <th>2004-10-15</th>
      <td>2004-10-15</td>
      <td>144.950</td>
      <td>145.50</td>
      <td>141.95</td>
      <td>144.110</td>
      <td>72.278116</td>
      <td>13194700.0</td>
    </tr>
    <tr>
      <th>2004-10-18</th>
      <td>2004-10-18</td>
      <td>143.120</td>
      <td>149.20</td>
      <td>141.21</td>
      <td>149.160</td>
      <td>74.810934</td>
      <td>14036300.0</td>
    </tr>
    <tr>
      <th>2004-10-19</th>
      <td>2004-10-19</td>
      <td>150.500</td>
      <td>152.40</td>
      <td>147.35</td>
      <td>147.940</td>
      <td>74.199045</td>
      <td>18109800.0</td>
    </tr>
    <tr>
      <th>2004-10-20</th>
      <td>2004-10-20</td>
      <td>147.940</td>
      <td>148.99</td>
      <td>139.60</td>
      <td>140.490</td>
      <td>70.462511</td>
      <td>22722600.0</td>
    </tr>
    <tr>
      <th>2004-10-21</th>
      <td>2004-10-21</td>
      <td>144.130</td>
      <td>150.13</td>
      <td>141.62</td>
      <td>149.380</td>
      <td>74.921275</td>
      <td>29149800.0</td>
    </tr>
    <tr>
      <th>2004-10-22</th>
      <td>2004-10-22</td>
      <td>170.435</td>
      <td>180.17</td>
      <td>164.08</td>
      <td>172.430</td>
      <td>86.481962</td>
      <td>73710000.0</td>
    </tr>
    <tr>
      <th>2004-10-25</th>
      <td>2004-10-25</td>
      <td>176.280</td>
      <td>194.43</td>
      <td>172.55</td>
      <td>187.400</td>
      <td>93.990139</td>
      <td>65462800.0</td>
    </tr>
    <tr>
      <th>2004-10-26</th>
      <td>2004-10-26</td>
      <td>186.449</td>
      <td>192.64</td>
      <td>180.00</td>
      <td>181.800</td>
      <td>91.181468</td>
      <td>44569500.0</td>
    </tr>
    <tr>
      <th>2004-10-27</th>
      <td>2004-10-27</td>
      <td>182.509</td>
      <td>189.52</td>
      <td>181.77</td>
      <td>185.970</td>
      <td>93.272925</td>
      <td>26686200.0</td>
    </tr>
    <tr>
      <th>2004-10-28</th>
      <td>2004-10-28</td>
      <td>186.630</td>
      <td>194.39</td>
      <td>185.60</td>
      <td>193.300</td>
      <td>96.949273</td>
      <td>29663900.0</td>
    </tr>
  </tbody>
</table>
</div>




```
plt.figure(figsize=(16,8))
plt.plot(df['Date'], df['Adj. Close'], label='Close Price history')
```




    [<matplotlib.lines.Line2D at 0x7f6e00e4acc0>]




![png](images/Stock%20Prediction%20Project_42_1.png)



```
# Creating dataframe with date and the target variable

data = df.sort_index(ascending=True, axis=0)
new_data = pd.DataFrame(index=range(0, len(df)), columns=['Date', 'Adj. Close'])

for i in range(0, len(data)):
  new_data['Date'][i] = data['Date'][i]
  new_data['Adj. Close'][i] = data['Adj. Close'][i]
```


```
# Train-test split

train = new_data[:2600]
test = new_data[2600:]
```


```
new_data.shape, train.shape, test.shape
```




    ((3424, 2), (2600, 2), (824, 2))




```
num = test.shape[0]
```


```
train['Date'].min(), train['Date'].max(), test['Date'].min(), test['Date'].max()
```




    (Timestamp('2004-08-19 00:00:00'),
     Timestamp('2014-12-15 00:00:00'),
     Timestamp('2014-12-16 00:00:00'),
     Timestamp('2018-03-27 00:00:00'))




```
# Making predictions

preds = []
for i in range(0, num):
  a = train['Adj. Close'][len(train)-924+i:].sum() + sum(preds)
  b = a/num
  preds.append(b)
```


```
len(preds)
```




    824




```
# Measure accuracy with rmse (Root Mean Squared Error)

rms=np.sqrt(np.mean(np.power((np.array(test['Adj. Close'])-preds),2)))

print(rms)
```

    264.46002931639754



```
test['Predictions'] = 0
test['Predictions'] = preds
plt.plot(train['Adj. Close'])
plt.plot(test[['Adj. Close', 'Predictions']])
```

    /usr/local/lib/python3.6/dist-packages/ipykernel_launcher.py:1: SettingWithCopyWarning:
    
    
    A value is trying to be set on a copy of a slice from a DataFrame.
    Try using .loc[row_indexer,col_indexer] = value instead
    
    See the caveats in the documentation: http://pandas.pydata.org/pandas-docs/stable/indexing.html#indexing-view-versus-copy
    
    /usr/local/lib/python3.6/dist-packages/ipykernel_launcher.py:2: SettingWithCopyWarning:
    
    
    A value is trying to be set on a copy of a slice from a DataFrame.
    Try using .loc[row_indexer,col_indexer] = value instead
    
    See the caveats in the documentation: http://pandas.pydata.org/pandas-docs/stable/indexing.html#indexing-view-versus-copy
    





    [<matplotlib.lines.Line2D at 0x7f6dee712940>,
     <matplotlib.lines.Line2D at 0x7f6dee712b70>]




![png](images/Stock%20Prediction%20Project_51_2.png)


# Simple Linear Regression


```
lr_data = goog_data
```


```
lr_data.head(50)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Date</th>
      <th>Open</th>
      <th>High</th>
      <th>Low</th>
      <th>Close</th>
      <th>Adj. Close</th>
      <th>Volume</th>
    </tr>
    <tr>
      <th>Date</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>2004-08-19</th>
      <td>2004-08-19</td>
      <td>100.010</td>
      <td>104.06</td>
      <td>95.96</td>
      <td>100.335</td>
      <td>50.322842</td>
      <td>44659000.0</td>
    </tr>
    <tr>
      <th>2004-08-20</th>
      <td>2004-08-20</td>
      <td>101.010</td>
      <td>109.08</td>
      <td>100.50</td>
      <td>108.310</td>
      <td>54.322689</td>
      <td>22834300.0</td>
    </tr>
    <tr>
      <th>2004-08-23</th>
      <td>2004-08-23</td>
      <td>110.760</td>
      <td>113.48</td>
      <td>109.05</td>
      <td>109.400</td>
      <td>54.869377</td>
      <td>18256100.0</td>
    </tr>
    <tr>
      <th>2004-08-24</th>
      <td>2004-08-24</td>
      <td>111.240</td>
      <td>111.60</td>
      <td>103.57</td>
      <td>104.870</td>
      <td>52.597363</td>
      <td>15247300.0</td>
    </tr>
    <tr>
      <th>2004-08-25</th>
      <td>2004-08-25</td>
      <td>104.760</td>
      <td>108.00</td>
      <td>103.88</td>
      <td>106.000</td>
      <td>53.164113</td>
      <td>9188600.0</td>
    </tr>
    <tr>
      <th>2004-08-26</th>
      <td>2004-08-26</td>
      <td>104.950</td>
      <td>107.95</td>
      <td>104.66</td>
      <td>107.910</td>
      <td>54.122070</td>
      <td>7094800.0</td>
    </tr>
    <tr>
      <th>2004-08-27</th>
      <td>2004-08-27</td>
      <td>108.100</td>
      <td>108.62</td>
      <td>105.69</td>
      <td>106.150</td>
      <td>53.239345</td>
      <td>6211700.0</td>
    </tr>
    <tr>
      <th>2004-08-30</th>
      <td>2004-08-30</td>
      <td>105.280</td>
      <td>105.49</td>
      <td>102.01</td>
      <td>102.010</td>
      <td>51.162935</td>
      <td>5196700.0</td>
    </tr>
    <tr>
      <th>2004-08-31</th>
      <td>2004-08-31</td>
      <td>102.320</td>
      <td>103.71</td>
      <td>102.16</td>
      <td>102.370</td>
      <td>51.343492</td>
      <td>4917800.0</td>
    </tr>
    <tr>
      <th>2004-09-01</th>
      <td>2004-09-01</td>
      <td>102.700</td>
      <td>102.97</td>
      <td>99.67</td>
      <td>100.250</td>
      <td>50.280210</td>
      <td>9138200.0</td>
    </tr>
    <tr>
      <th>2004-09-02</th>
      <td>2004-09-02</td>
      <td>99.090</td>
      <td>102.37</td>
      <td>98.94</td>
      <td>101.510</td>
      <td>50.912161</td>
      <td>15118600.0</td>
    </tr>
    <tr>
      <th>2004-09-03</th>
      <td>2004-09-03</td>
      <td>100.950</td>
      <td>101.74</td>
      <td>99.32</td>
      <td>100.010</td>
      <td>50.159839</td>
      <td>5152400.0</td>
    </tr>
    <tr>
      <th>2004-09-07</th>
      <td>2004-09-07</td>
      <td>101.010</td>
      <td>102.00</td>
      <td>99.61</td>
      <td>101.580</td>
      <td>50.947269</td>
      <td>5847500.0</td>
    </tr>
    <tr>
      <th>2004-09-08</th>
      <td>2004-09-08</td>
      <td>100.740</td>
      <td>103.03</td>
      <td>100.50</td>
      <td>102.300</td>
      <td>51.308384</td>
      <td>4985600.0</td>
    </tr>
    <tr>
      <th>2004-09-09</th>
      <td>2004-09-09</td>
      <td>102.500</td>
      <td>102.71</td>
      <td>101.00</td>
      <td>102.310</td>
      <td>51.313400</td>
      <td>4061700.0</td>
    </tr>
    <tr>
      <th>2004-09-10</th>
      <td>2004-09-10</td>
      <td>101.470</td>
      <td>106.56</td>
      <td>101.30</td>
      <td>105.330</td>
      <td>52.828075</td>
      <td>8698800.0</td>
    </tr>
    <tr>
      <th>2004-09-13</th>
      <td>2004-09-13</td>
      <td>106.630</td>
      <td>108.41</td>
      <td>106.46</td>
      <td>107.500</td>
      <td>53.916435</td>
      <td>7844100.0</td>
    </tr>
    <tr>
      <th>2004-09-14</th>
      <td>2004-09-14</td>
      <td>107.440</td>
      <td>112.00</td>
      <td>106.79</td>
      <td>111.490</td>
      <td>55.917612</td>
      <td>10828900.0</td>
    </tr>
    <tr>
      <th>2004-09-15</th>
      <td>2004-09-15</td>
      <td>110.560</td>
      <td>114.23</td>
      <td>110.20</td>
      <td>112.000</td>
      <td>56.173402</td>
      <td>10713000.0</td>
    </tr>
    <tr>
      <th>2004-09-16</th>
      <td>2004-09-16</td>
      <td>112.340</td>
      <td>115.80</td>
      <td>111.65</td>
      <td>113.970</td>
      <td>57.161452</td>
      <td>9266300.0</td>
    </tr>
    <tr>
      <th>2004-09-17</th>
      <td>2004-09-17</td>
      <td>114.420</td>
      <td>117.49</td>
      <td>113.55</td>
      <td>117.490</td>
      <td>58.926902</td>
      <td>9472500.0</td>
    </tr>
    <tr>
      <th>2004-09-20</th>
      <td>2004-09-20</td>
      <td>116.950</td>
      <td>121.60</td>
      <td>116.77</td>
      <td>119.360</td>
      <td>59.864797</td>
      <td>10628700.0</td>
    </tr>
    <tr>
      <th>2004-09-21</th>
      <td>2004-09-21</td>
      <td>120.200</td>
      <td>120.42</td>
      <td>117.51</td>
      <td>117.840</td>
      <td>59.102444</td>
      <td>7228700.0</td>
    </tr>
    <tr>
      <th>2004-09-22</th>
      <td>2004-09-22</td>
      <td>117.450</td>
      <td>119.67</td>
      <td>116.81</td>
      <td>118.380</td>
      <td>59.373280</td>
      <td>7581200.0</td>
    </tr>
    <tr>
      <th>2004-09-23</th>
      <td>2004-09-23</td>
      <td>118.840</td>
      <td>122.63</td>
      <td>117.02</td>
      <td>120.820</td>
      <td>60.597057</td>
      <td>8535600.0</td>
    </tr>
    <tr>
      <th>2004-09-24</th>
      <td>2004-09-24</td>
      <td>120.970</td>
      <td>124.10</td>
      <td>119.76</td>
      <td>119.830</td>
      <td>60.100525</td>
      <td>9123400.0</td>
    </tr>
    <tr>
      <th>2004-09-27</th>
      <td>2004-09-27</td>
      <td>119.560</td>
      <td>120.88</td>
      <td>117.80</td>
      <td>118.260</td>
      <td>59.313094</td>
      <td>7066100.0</td>
    </tr>
    <tr>
      <th>2004-09-28</th>
      <td>2004-09-28</td>
      <td>121.150</td>
      <td>127.40</td>
      <td>120.21</td>
      <td>126.860</td>
      <td>63.626409</td>
      <td>16929000.0</td>
    </tr>
    <tr>
      <th>2004-09-29</th>
      <td>2004-09-29</td>
      <td>126.530</td>
      <td>135.02</td>
      <td>126.23</td>
      <td>131.080</td>
      <td>65.742942</td>
      <td>30516400.0</td>
    </tr>
    <tr>
      <th>2004-09-30</th>
      <td>2004-09-30</td>
      <td>129.899</td>
      <td>132.30</td>
      <td>129.00</td>
      <td>129.600</td>
      <td>65.000651</td>
      <td>13758000.0</td>
    </tr>
    <tr>
      <th>2004-10-01</th>
      <td>2004-10-01</td>
      <td>130.800</td>
      <td>134.24</td>
      <td>128.90</td>
      <td>132.580</td>
      <td>66.495265</td>
      <td>15124800.0</td>
    </tr>
    <tr>
      <th>2004-10-04</th>
      <td>2004-10-04</td>
      <td>135.275</td>
      <td>136.87</td>
      <td>134.03</td>
      <td>135.060</td>
      <td>67.739104</td>
      <td>13022700.0</td>
    </tr>
    <tr>
      <th>2004-10-05</th>
      <td>2004-10-05</td>
      <td>134.660</td>
      <td>138.53</td>
      <td>132.24</td>
      <td>138.370</td>
      <td>69.399229</td>
      <td>14973200.0</td>
    </tr>
    <tr>
      <th>2004-10-06</th>
      <td>2004-10-06</td>
      <td>137.670</td>
      <td>138.45</td>
      <td>136.00</td>
      <td>137.080</td>
      <td>68.752232</td>
      <td>13381400.0</td>
    </tr>
    <tr>
      <th>2004-10-07</th>
      <td>2004-10-07</td>
      <td>136.560</td>
      <td>139.88</td>
      <td>136.55</td>
      <td>138.850</td>
      <td>69.639972</td>
      <td>14115000.0</td>
    </tr>
    <tr>
      <th>2004-10-08</th>
      <td>2004-10-08</td>
      <td>138.730</td>
      <td>139.68</td>
      <td>137.02</td>
      <td>137.730</td>
      <td>69.078238</td>
      <td>11069500.0</td>
    </tr>
    <tr>
      <th>2004-10-11</th>
      <td>2004-10-11</td>
      <td>137.010</td>
      <td>138.86</td>
      <td>133.85</td>
      <td>135.260</td>
      <td>67.839414</td>
      <td>10472100.0</td>
    </tr>
    <tr>
      <th>2004-10-12</th>
      <td>2004-10-12</td>
      <td>134.490</td>
      <td>137.61</td>
      <td>133.40</td>
      <td>137.400</td>
      <td>68.912727</td>
      <td>11665500.0</td>
    </tr>
    <tr>
      <th>2004-10-13</th>
      <td>2004-10-13</td>
      <td>143.240</td>
      <td>143.55</td>
      <td>140.08</td>
      <td>140.900</td>
      <td>70.668146</td>
      <td>19766200.0</td>
    </tr>
    <tr>
      <th>2004-10-14</th>
      <td>2004-10-14</td>
      <td>141.020</td>
      <td>142.38</td>
      <td>138.56</td>
      <td>142.000</td>
      <td>71.219849</td>
      <td>10442100.0</td>
    </tr>
    <tr>
      <th>2004-10-15</th>
      <td>2004-10-15</td>
      <td>144.950</td>
      <td>145.50</td>
      <td>141.95</td>
      <td>144.110</td>
      <td>72.278116</td>
      <td>13194700.0</td>
    </tr>
    <tr>
      <th>2004-10-18</th>
      <td>2004-10-18</td>
      <td>143.120</td>
      <td>149.20</td>
      <td>141.21</td>
      <td>149.160</td>
      <td>74.810934</td>
      <td>14036300.0</td>
    </tr>
    <tr>
      <th>2004-10-19</th>
      <td>2004-10-19</td>
      <td>150.500</td>
      <td>152.40</td>
      <td>147.35</td>
      <td>147.940</td>
      <td>74.199045</td>
      <td>18109800.0</td>
    </tr>
    <tr>
      <th>2004-10-20</th>
      <td>2004-10-20</td>
      <td>147.940</td>
      <td>148.99</td>
      <td>139.60</td>
      <td>140.490</td>
      <td>70.462511</td>
      <td>22722600.0</td>
    </tr>
    <tr>
      <th>2004-10-21</th>
      <td>2004-10-21</td>
      <td>144.130</td>
      <td>150.13</td>
      <td>141.62</td>
      <td>149.380</td>
      <td>74.921275</td>
      <td>29149800.0</td>
    </tr>
    <tr>
      <th>2004-10-22</th>
      <td>2004-10-22</td>
      <td>170.435</td>
      <td>180.17</td>
      <td>164.08</td>
      <td>172.430</td>
      <td>86.481962</td>
      <td>73710000.0</td>
    </tr>
    <tr>
      <th>2004-10-25</th>
      <td>2004-10-25</td>
      <td>176.280</td>
      <td>194.43</td>
      <td>172.55</td>
      <td>187.400</td>
      <td>93.990139</td>
      <td>65462800.0</td>
    </tr>
    <tr>
      <th>2004-10-26</th>
      <td>2004-10-26</td>
      <td>186.449</td>
      <td>192.64</td>
      <td>180.00</td>
      <td>181.800</td>
      <td>91.181468</td>
      <td>44569500.0</td>
    </tr>
    <tr>
      <th>2004-10-27</th>
      <td>2004-10-27</td>
      <td>182.509</td>
      <td>189.52</td>
      <td>181.77</td>
      <td>185.970</td>
      <td>93.272925</td>
      <td>26686200.0</td>
    </tr>
    <tr>
      <th>2004-10-28</th>
      <td>2004-10-28</td>
      <td>186.630</td>
      <td>194.39</td>
      <td>185.60</td>
      <td>193.300</td>
      <td>96.949273</td>
      <td>29663900.0</td>
    </tr>
  </tbody>
</table>
</div>




```
# We'll create a separate dataset so that new features don't mess up the original data.


lr_data['Date'] = pd.to_datetime(lr_data.Date, format='%Y-%m-%d')
lr_data.index = lr_data['Date']
```


```
lr_data = lr_data.sort_index(ascending=True, axis=0)
```


```
new_data = pd.DataFrame(index=range(0, len(lr_data)), columns=['Date', 'Adj. Close'])
for i in range(0,len(data)):
    new_data['Date'][i] = lr_data['Date'][i]
    new_data['Adj. Close'][i] = lr_data['Adj. Close'][i]


```


```
new_data.head(50)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Date</th>
      <th>Adj. Close</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>2004-08-19 00:00:00</td>
      <td>50.3228</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2004-08-20 00:00:00</td>
      <td>54.3227</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2004-08-23 00:00:00</td>
      <td>54.8694</td>
    </tr>
    <tr>
      <th>3</th>
      <td>2004-08-24 00:00:00</td>
      <td>52.5974</td>
    </tr>
    <tr>
      <th>4</th>
      <td>2004-08-25 00:00:00</td>
      <td>53.1641</td>
    </tr>
    <tr>
      <th>5</th>
      <td>2004-08-26 00:00:00</td>
      <td>54.1221</td>
    </tr>
    <tr>
      <th>6</th>
      <td>2004-08-27 00:00:00</td>
      <td>53.2393</td>
    </tr>
    <tr>
      <th>7</th>
      <td>2004-08-30 00:00:00</td>
      <td>51.1629</td>
    </tr>
    <tr>
      <th>8</th>
      <td>2004-08-31 00:00:00</td>
      <td>51.3435</td>
    </tr>
    <tr>
      <th>9</th>
      <td>2004-09-01 00:00:00</td>
      <td>50.2802</td>
    </tr>
    <tr>
      <th>10</th>
      <td>2004-09-02 00:00:00</td>
      <td>50.9122</td>
    </tr>
    <tr>
      <th>11</th>
      <td>2004-09-03 00:00:00</td>
      <td>50.1598</td>
    </tr>
    <tr>
      <th>12</th>
      <td>2004-09-07 00:00:00</td>
      <td>50.9473</td>
    </tr>
    <tr>
      <th>13</th>
      <td>2004-09-08 00:00:00</td>
      <td>51.3084</td>
    </tr>
    <tr>
      <th>14</th>
      <td>2004-09-09 00:00:00</td>
      <td>51.3134</td>
    </tr>
    <tr>
      <th>15</th>
      <td>2004-09-10 00:00:00</td>
      <td>52.8281</td>
    </tr>
    <tr>
      <th>16</th>
      <td>2004-09-13 00:00:00</td>
      <td>53.9164</td>
    </tr>
    <tr>
      <th>17</th>
      <td>2004-09-14 00:00:00</td>
      <td>55.9176</td>
    </tr>
    <tr>
      <th>18</th>
      <td>2004-09-15 00:00:00</td>
      <td>56.1734</td>
    </tr>
    <tr>
      <th>19</th>
      <td>2004-09-16 00:00:00</td>
      <td>57.1615</td>
    </tr>
    <tr>
      <th>20</th>
      <td>2004-09-17 00:00:00</td>
      <td>58.9269</td>
    </tr>
    <tr>
      <th>21</th>
      <td>2004-09-20 00:00:00</td>
      <td>59.8648</td>
    </tr>
    <tr>
      <th>22</th>
      <td>2004-09-21 00:00:00</td>
      <td>59.1024</td>
    </tr>
    <tr>
      <th>23</th>
      <td>2004-09-22 00:00:00</td>
      <td>59.3733</td>
    </tr>
    <tr>
      <th>24</th>
      <td>2004-09-23 00:00:00</td>
      <td>60.5971</td>
    </tr>
    <tr>
      <th>25</th>
      <td>2004-09-24 00:00:00</td>
      <td>60.1005</td>
    </tr>
    <tr>
      <th>26</th>
      <td>2004-09-27 00:00:00</td>
      <td>59.3131</td>
    </tr>
    <tr>
      <th>27</th>
      <td>2004-09-28 00:00:00</td>
      <td>63.6264</td>
    </tr>
    <tr>
      <th>28</th>
      <td>2004-09-29 00:00:00</td>
      <td>65.7429</td>
    </tr>
    <tr>
      <th>29</th>
      <td>2004-09-30 00:00:00</td>
      <td>65.0007</td>
    </tr>
    <tr>
      <th>30</th>
      <td>2004-10-01 00:00:00</td>
      <td>66.4953</td>
    </tr>
    <tr>
      <th>31</th>
      <td>2004-10-04 00:00:00</td>
      <td>67.7391</td>
    </tr>
    <tr>
      <th>32</th>
      <td>2004-10-05 00:00:00</td>
      <td>69.3992</td>
    </tr>
    <tr>
      <th>33</th>
      <td>2004-10-06 00:00:00</td>
      <td>68.7522</td>
    </tr>
    <tr>
      <th>34</th>
      <td>2004-10-07 00:00:00</td>
      <td>69.64</td>
    </tr>
    <tr>
      <th>35</th>
      <td>2004-10-08 00:00:00</td>
      <td>69.0782</td>
    </tr>
    <tr>
      <th>36</th>
      <td>2004-10-11 00:00:00</td>
      <td>67.8394</td>
    </tr>
    <tr>
      <th>37</th>
      <td>2004-10-12 00:00:00</td>
      <td>68.9127</td>
    </tr>
    <tr>
      <th>38</th>
      <td>2004-10-13 00:00:00</td>
      <td>70.6681</td>
    </tr>
    <tr>
      <th>39</th>
      <td>2004-10-14 00:00:00</td>
      <td>71.2198</td>
    </tr>
    <tr>
      <th>40</th>
      <td>2004-10-15 00:00:00</td>
      <td>72.2781</td>
    </tr>
    <tr>
      <th>41</th>
      <td>2004-10-18 00:00:00</td>
      <td>74.8109</td>
    </tr>
    <tr>
      <th>42</th>
      <td>2004-10-19 00:00:00</td>
      <td>74.199</td>
    </tr>
    <tr>
      <th>43</th>
      <td>2004-10-20 00:00:00</td>
      <td>70.4625</td>
    </tr>
    <tr>
      <th>44</th>
      <td>2004-10-21 00:00:00</td>
      <td>74.9213</td>
    </tr>
    <tr>
      <th>45</th>
      <td>2004-10-22 00:00:00</td>
      <td>86.482</td>
    </tr>
    <tr>
      <th>46</th>
      <td>2004-10-25 00:00:00</td>
      <td>93.9901</td>
    </tr>
    <tr>
      <th>47</th>
      <td>2004-10-26 00:00:00</td>
      <td>91.1815</td>
    </tr>
    <tr>
      <th>48</th>
      <td>2004-10-27 00:00:00</td>
      <td>93.2729</td>
    </tr>
    <tr>
      <th>49</th>
      <td>2004-10-28 00:00:00</td>
      <td>96.9493</td>
    </tr>
  </tbody>
</table>
</div>




```
!pip install fastai==0.7.0
```


```
from fastai.structured import add_datepart


```


```
add_datepart(new_data, 'Date')
new_data.drop('Elapsed', axis=1, inplace=True)
```


```
new_data.head(50)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Adj. Close</th>
      <th>Year</th>
      <th>Month</th>
      <th>Week</th>
      <th>Day</th>
      <th>Dayofweek</th>
      <th>Dayofyear</th>
      <th>Is_month_end</th>
      <th>Is_month_start</th>
      <th>Is_quarter_end</th>
      <th>Is_quarter_start</th>
      <th>Is_year_end</th>
      <th>Is_year_start</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>50.3228</td>
      <td>2004</td>
      <td>8</td>
      <td>34</td>
      <td>19</td>
      <td>3</td>
      <td>232</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
    </tr>
    <tr>
      <th>1</th>
      <td>54.3227</td>
      <td>2004</td>
      <td>8</td>
      <td>34</td>
      <td>20</td>
      <td>4</td>
      <td>233</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
    </tr>
    <tr>
      <th>2</th>
      <td>54.8694</td>
      <td>2004</td>
      <td>8</td>
      <td>35</td>
      <td>23</td>
      <td>0</td>
      <td>236</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
    </tr>
    <tr>
      <th>3</th>
      <td>52.5974</td>
      <td>2004</td>
      <td>8</td>
      <td>35</td>
      <td>24</td>
      <td>1</td>
      <td>237</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
    </tr>
    <tr>
      <th>4</th>
      <td>53.1641</td>
      <td>2004</td>
      <td>8</td>
      <td>35</td>
      <td>25</td>
      <td>2</td>
      <td>238</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
    </tr>
    <tr>
      <th>5</th>
      <td>54.1221</td>
      <td>2004</td>
      <td>8</td>
      <td>35</td>
      <td>26</td>
      <td>3</td>
      <td>239</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
    </tr>
    <tr>
      <th>6</th>
      <td>53.2393</td>
      <td>2004</td>
      <td>8</td>
      <td>35</td>
      <td>27</td>
      <td>4</td>
      <td>240</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
    </tr>
    <tr>
      <th>7</th>
      <td>51.1629</td>
      <td>2004</td>
      <td>8</td>
      <td>36</td>
      <td>30</td>
      <td>0</td>
      <td>243</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
    </tr>
    <tr>
      <th>8</th>
      <td>51.3435</td>
      <td>2004</td>
      <td>8</td>
      <td>36</td>
      <td>31</td>
      <td>1</td>
      <td>244</td>
      <td>True</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
    </tr>
    <tr>
      <th>9</th>
      <td>50.2802</td>
      <td>2004</td>
      <td>9</td>
      <td>36</td>
      <td>1</td>
      <td>2</td>
      <td>245</td>
      <td>False</td>
      <td>True</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
    </tr>
    <tr>
      <th>10</th>
      <td>50.9122</td>
      <td>2004</td>
      <td>9</td>
      <td>36</td>
      <td>2</td>
      <td>3</td>
      <td>246</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
    </tr>
    <tr>
      <th>11</th>
      <td>50.1598</td>
      <td>2004</td>
      <td>9</td>
      <td>36</td>
      <td>3</td>
      <td>4</td>
      <td>247</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
    </tr>
    <tr>
      <th>12</th>
      <td>50.9473</td>
      <td>2004</td>
      <td>9</td>
      <td>37</td>
      <td>7</td>
      <td>1</td>
      <td>251</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
    </tr>
    <tr>
      <th>13</th>
      <td>51.3084</td>
      <td>2004</td>
      <td>9</td>
      <td>37</td>
      <td>8</td>
      <td>2</td>
      <td>252</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
    </tr>
    <tr>
      <th>14</th>
      <td>51.3134</td>
      <td>2004</td>
      <td>9</td>
      <td>37</td>
      <td>9</td>
      <td>3</td>
      <td>253</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
    </tr>
    <tr>
      <th>15</th>
      <td>52.8281</td>
      <td>2004</td>
      <td>9</td>
      <td>37</td>
      <td>10</td>
      <td>4</td>
      <td>254</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
    </tr>
    <tr>
      <th>16</th>
      <td>53.9164</td>
      <td>2004</td>
      <td>9</td>
      <td>38</td>
      <td>13</td>
      <td>0</td>
      <td>257</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
    </tr>
    <tr>
      <th>17</th>
      <td>55.9176</td>
      <td>2004</td>
      <td>9</td>
      <td>38</td>
      <td>14</td>
      <td>1</td>
      <td>258</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
    </tr>
    <tr>
      <th>18</th>
      <td>56.1734</td>
      <td>2004</td>
      <td>9</td>
      <td>38</td>
      <td>15</td>
      <td>2</td>
      <td>259</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
    </tr>
    <tr>
      <th>19</th>
      <td>57.1615</td>
      <td>2004</td>
      <td>9</td>
      <td>38</td>
      <td>16</td>
      <td>3</td>
      <td>260</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
    </tr>
    <tr>
      <th>20</th>
      <td>58.9269</td>
      <td>2004</td>
      <td>9</td>
      <td>38</td>
      <td>17</td>
      <td>4</td>
      <td>261</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
    </tr>
    <tr>
      <th>21</th>
      <td>59.8648</td>
      <td>2004</td>
      <td>9</td>
      <td>39</td>
      <td>20</td>
      <td>0</td>
      <td>264</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
    </tr>
    <tr>
      <th>22</th>
      <td>59.1024</td>
      <td>2004</td>
      <td>9</td>
      <td>39</td>
      <td>21</td>
      <td>1</td>
      <td>265</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
    </tr>
    <tr>
      <th>23</th>
      <td>59.3733</td>
      <td>2004</td>
      <td>9</td>
      <td>39</td>
      <td>22</td>
      <td>2</td>
      <td>266</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
    </tr>
    <tr>
      <th>24</th>
      <td>60.5971</td>
      <td>2004</td>
      <td>9</td>
      <td>39</td>
      <td>23</td>
      <td>3</td>
      <td>267</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
    </tr>
    <tr>
      <th>25</th>
      <td>60.1005</td>
      <td>2004</td>
      <td>9</td>
      <td>39</td>
      <td>24</td>
      <td>4</td>
      <td>268</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
    </tr>
    <tr>
      <th>26</th>
      <td>59.3131</td>
      <td>2004</td>
      <td>9</td>
      <td>40</td>
      <td>27</td>
      <td>0</td>
      <td>271</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
    </tr>
    <tr>
      <th>27</th>
      <td>63.6264</td>
      <td>2004</td>
      <td>9</td>
      <td>40</td>
      <td>28</td>
      <td>1</td>
      <td>272</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
    </tr>
    <tr>
      <th>28</th>
      <td>65.7429</td>
      <td>2004</td>
      <td>9</td>
      <td>40</td>
      <td>29</td>
      <td>2</td>
      <td>273</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
    </tr>
    <tr>
      <th>29</th>
      <td>65.0007</td>
      <td>2004</td>
      <td>9</td>
      <td>40</td>
      <td>30</td>
      <td>3</td>
      <td>274</td>
      <td>True</td>
      <td>False</td>
      <td>True</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
    </tr>
    <tr>
      <th>30</th>
      <td>66.4953</td>
      <td>2004</td>
      <td>10</td>
      <td>40</td>
      <td>1</td>
      <td>4</td>
      <td>275</td>
      <td>False</td>
      <td>True</td>
      <td>False</td>
      <td>True</td>
      <td>False</td>
      <td>False</td>
    </tr>
    <tr>
      <th>31</th>
      <td>67.7391</td>
      <td>2004</td>
      <td>10</td>
      <td>41</td>
      <td>4</td>
      <td>0</td>
      <td>278</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
    </tr>
    <tr>
      <th>32</th>
      <td>69.3992</td>
      <td>2004</td>
      <td>10</td>
      <td>41</td>
      <td>5</td>
      <td>1</td>
      <td>279</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
    </tr>
    <tr>
      <th>33</th>
      <td>68.7522</td>
      <td>2004</td>
      <td>10</td>
      <td>41</td>
      <td>6</td>
      <td>2</td>
      <td>280</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
    </tr>
    <tr>
      <th>34</th>
      <td>69.64</td>
      <td>2004</td>
      <td>10</td>
      <td>41</td>
      <td>7</td>
      <td>3</td>
      <td>281</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
    </tr>
    <tr>
      <th>35</th>
      <td>69.0782</td>
      <td>2004</td>
      <td>10</td>
      <td>41</td>
      <td>8</td>
      <td>4</td>
      <td>282</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
    </tr>
    <tr>
      <th>36</th>
      <td>67.8394</td>
      <td>2004</td>
      <td>10</td>
      <td>42</td>
      <td>11</td>
      <td>0</td>
      <td>285</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
    </tr>
    <tr>
      <th>37</th>
      <td>68.9127</td>
      <td>2004</td>
      <td>10</td>
      <td>42</td>
      <td>12</td>
      <td>1</td>
      <td>286</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
    </tr>
    <tr>
      <th>38</th>
      <td>70.6681</td>
      <td>2004</td>
      <td>10</td>
      <td>42</td>
      <td>13</td>
      <td>2</td>
      <td>287</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
    </tr>
    <tr>
      <th>39</th>
      <td>71.2198</td>
      <td>2004</td>
      <td>10</td>
      <td>42</td>
      <td>14</td>
      <td>3</td>
      <td>288</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
    </tr>
    <tr>
      <th>40</th>
      <td>72.2781</td>
      <td>2004</td>
      <td>10</td>
      <td>42</td>
      <td>15</td>
      <td>4</td>
      <td>289</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
    </tr>
    <tr>
      <th>41</th>
      <td>74.8109</td>
      <td>2004</td>
      <td>10</td>
      <td>43</td>
      <td>18</td>
      <td>0</td>
      <td>292</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
    </tr>
    <tr>
      <th>42</th>
      <td>74.199</td>
      <td>2004</td>
      <td>10</td>
      <td>43</td>
      <td>19</td>
      <td>1</td>
      <td>293</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
    </tr>
    <tr>
      <th>43</th>
      <td>70.4625</td>
      <td>2004</td>
      <td>10</td>
      <td>43</td>
      <td>20</td>
      <td>2</td>
      <td>294</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
    </tr>
    <tr>
      <th>44</th>
      <td>74.9213</td>
      <td>2004</td>
      <td>10</td>
      <td>43</td>
      <td>21</td>
      <td>3</td>
      <td>295</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
    </tr>
    <tr>
      <th>45</th>
      <td>86.482</td>
      <td>2004</td>
      <td>10</td>
      <td>43</td>
      <td>22</td>
      <td>4</td>
      <td>296</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
    </tr>
    <tr>
      <th>46</th>
      <td>93.9901</td>
      <td>2004</td>
      <td>10</td>
      <td>44</td>
      <td>25</td>
      <td>0</td>
      <td>299</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
    </tr>
    <tr>
      <th>47</th>
      <td>91.1815</td>
      <td>2004</td>
      <td>10</td>
      <td>44</td>
      <td>26</td>
      <td>1</td>
      <td>300</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
    </tr>
    <tr>
      <th>48</th>
      <td>93.2729</td>
      <td>2004</td>
      <td>10</td>
      <td>44</td>
      <td>27</td>
      <td>2</td>
      <td>301</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
    </tr>
    <tr>
      <th>49</th>
      <td>96.9493</td>
      <td>2004</td>
      <td>10</td>
      <td>44</td>
      <td>28</td>
      <td>3</td>
      <td>302</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
    </tr>
  </tbody>
</table>
</div>




```
# Train-test split

train = new_data[:2600]
test = new_data[2600:]

x_train = train.drop('Adj. Close', axis=1)
y_train = train['Adj. Close']
x_test = test.drop('Adj. Close', axis=1)
y_test = test['Adj. Close']
```


```
x_train
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Year</th>
      <th>Month</th>
      <th>Week</th>
      <th>Day</th>
      <th>Dayofweek</th>
      <th>Dayofyear</th>
      <th>Is_month_end</th>
      <th>Is_month_start</th>
      <th>Is_quarter_end</th>
      <th>Is_quarter_start</th>
      <th>Is_year_end</th>
      <th>Is_year_start</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>2004</td>
      <td>8</td>
      <td>34</td>
      <td>19</td>
      <td>3</td>
      <td>232</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2004</td>
      <td>8</td>
      <td>34</td>
      <td>20</td>
      <td>4</td>
      <td>233</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2004</td>
      <td>8</td>
      <td>35</td>
      <td>23</td>
      <td>0</td>
      <td>236</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
    </tr>
    <tr>
      <th>3</th>
      <td>2004</td>
      <td>8</td>
      <td>35</td>
      <td>24</td>
      <td>1</td>
      <td>237</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
    </tr>
    <tr>
      <th>4</th>
      <td>2004</td>
      <td>8</td>
      <td>35</td>
      <td>25</td>
      <td>2</td>
      <td>238</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
    </tr>
    <tr>
      <th>5</th>
      <td>2004</td>
      <td>8</td>
      <td>35</td>
      <td>26</td>
      <td>3</td>
      <td>239</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
    </tr>
    <tr>
      <th>6</th>
      <td>2004</td>
      <td>8</td>
      <td>35</td>
      <td>27</td>
      <td>4</td>
      <td>240</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
    </tr>
    <tr>
      <th>7</th>
      <td>2004</td>
      <td>8</td>
      <td>36</td>
      <td>30</td>
      <td>0</td>
      <td>243</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
    </tr>
    <tr>
      <th>8</th>
      <td>2004</td>
      <td>8</td>
      <td>36</td>
      <td>31</td>
      <td>1</td>
      <td>244</td>
      <td>True</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
    </tr>
    <tr>
      <th>9</th>
      <td>2004</td>
      <td>9</td>
      <td>36</td>
      <td>1</td>
      <td>2</td>
      <td>245</td>
      <td>False</td>
      <td>True</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
    </tr>
    <tr>
      <th>10</th>
      <td>2004</td>
      <td>9</td>
      <td>36</td>
      <td>2</td>
      <td>3</td>
      <td>246</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
    </tr>
    <tr>
      <th>11</th>
      <td>2004</td>
      <td>9</td>
      <td>36</td>
      <td>3</td>
      <td>4</td>
      <td>247</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
    </tr>
    <tr>
      <th>12</th>
      <td>2004</td>
      <td>9</td>
      <td>37</td>
      <td>7</td>
      <td>1</td>
      <td>251</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
    </tr>
    <tr>
      <th>13</th>
      <td>2004</td>
      <td>9</td>
      <td>37</td>
      <td>8</td>
      <td>2</td>
      <td>252</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
    </tr>
    <tr>
      <th>14</th>
      <td>2004</td>
      <td>9</td>
      <td>37</td>
      <td>9</td>
      <td>3</td>
      <td>253</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
    </tr>
    <tr>
      <th>15</th>
      <td>2004</td>
      <td>9</td>
      <td>37</td>
      <td>10</td>
      <td>4</td>
      <td>254</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
    </tr>
    <tr>
      <th>16</th>
      <td>2004</td>
      <td>9</td>
      <td>38</td>
      <td>13</td>
      <td>0</td>
      <td>257</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
    </tr>
    <tr>
      <th>17</th>
      <td>2004</td>
      <td>9</td>
      <td>38</td>
      <td>14</td>
      <td>1</td>
      <td>258</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
    </tr>
    <tr>
      <th>18</th>
      <td>2004</td>
      <td>9</td>
      <td>38</td>
      <td>15</td>
      <td>2</td>
      <td>259</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
    </tr>
    <tr>
      <th>19</th>
      <td>2004</td>
      <td>9</td>
      <td>38</td>
      <td>16</td>
      <td>3</td>
      <td>260</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
    </tr>
    <tr>
      <th>20</th>
      <td>2004</td>
      <td>9</td>
      <td>38</td>
      <td>17</td>
      <td>4</td>
      <td>261</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
    </tr>
    <tr>
      <th>21</th>
      <td>2004</td>
      <td>9</td>
      <td>39</td>
      <td>20</td>
      <td>0</td>
      <td>264</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
    </tr>
    <tr>
      <th>22</th>
      <td>2004</td>
      <td>9</td>
      <td>39</td>
      <td>21</td>
      <td>1</td>
      <td>265</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
    </tr>
    <tr>
      <th>23</th>
      <td>2004</td>
      <td>9</td>
      <td>39</td>
      <td>22</td>
      <td>2</td>
      <td>266</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
    </tr>
    <tr>
      <th>24</th>
      <td>2004</td>
      <td>9</td>
      <td>39</td>
      <td>23</td>
      <td>3</td>
      <td>267</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
    </tr>
    <tr>
      <th>25</th>
      <td>2004</td>
      <td>9</td>
      <td>39</td>
      <td>24</td>
      <td>4</td>
      <td>268</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
    </tr>
    <tr>
      <th>26</th>
      <td>2004</td>
      <td>9</td>
      <td>40</td>
      <td>27</td>
      <td>0</td>
      <td>271</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
    </tr>
    <tr>
      <th>27</th>
      <td>2004</td>
      <td>9</td>
      <td>40</td>
      <td>28</td>
      <td>1</td>
      <td>272</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
    </tr>
    <tr>
      <th>28</th>
      <td>2004</td>
      <td>9</td>
      <td>40</td>
      <td>29</td>
      <td>2</td>
      <td>273</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
    </tr>
    <tr>
      <th>29</th>
      <td>2004</td>
      <td>9</td>
      <td>40</td>
      <td>30</td>
      <td>3</td>
      <td>274</td>
      <td>True</td>
      <td>False</td>
      <td>True</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>2570</th>
      <td>2014</td>
      <td>11</td>
      <td>45</td>
      <td>3</td>
      <td>0</td>
      <td>307</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
    </tr>
    <tr>
      <th>2571</th>
      <td>2014</td>
      <td>11</td>
      <td>45</td>
      <td>4</td>
      <td>1</td>
      <td>308</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
    </tr>
    <tr>
      <th>2572</th>
      <td>2014</td>
      <td>11</td>
      <td>45</td>
      <td>5</td>
      <td>2</td>
      <td>309</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
    </tr>
    <tr>
      <th>2573</th>
      <td>2014</td>
      <td>11</td>
      <td>45</td>
      <td>6</td>
      <td>3</td>
      <td>310</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
    </tr>
    <tr>
      <th>2574</th>
      <td>2014</td>
      <td>11</td>
      <td>45</td>
      <td>7</td>
      <td>4</td>
      <td>311</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
    </tr>
    <tr>
      <th>2575</th>
      <td>2014</td>
      <td>11</td>
      <td>46</td>
      <td>10</td>
      <td>0</td>
      <td>314</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
    </tr>
    <tr>
      <th>2576</th>
      <td>2014</td>
      <td>11</td>
      <td>46</td>
      <td>11</td>
      <td>1</td>
      <td>315</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
    </tr>
    <tr>
      <th>2577</th>
      <td>2014</td>
      <td>11</td>
      <td>46</td>
      <td>12</td>
      <td>2</td>
      <td>316</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
    </tr>
    <tr>
      <th>2578</th>
      <td>2014</td>
      <td>11</td>
      <td>46</td>
      <td>13</td>
      <td>3</td>
      <td>317</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
    </tr>
    <tr>
      <th>2579</th>
      <td>2014</td>
      <td>11</td>
      <td>46</td>
      <td>14</td>
      <td>4</td>
      <td>318</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
    </tr>
    <tr>
      <th>2580</th>
      <td>2014</td>
      <td>11</td>
      <td>47</td>
      <td>17</td>
      <td>0</td>
      <td>321</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
    </tr>
    <tr>
      <th>2581</th>
      <td>2014</td>
      <td>11</td>
      <td>47</td>
      <td>18</td>
      <td>1</td>
      <td>322</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
    </tr>
    <tr>
      <th>2582</th>
      <td>2014</td>
      <td>11</td>
      <td>47</td>
      <td>19</td>
      <td>2</td>
      <td>323</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
    </tr>
    <tr>
      <th>2583</th>
      <td>2014</td>
      <td>11</td>
      <td>47</td>
      <td>20</td>
      <td>3</td>
      <td>324</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
    </tr>
    <tr>
      <th>2584</th>
      <td>2014</td>
      <td>11</td>
      <td>47</td>
      <td>21</td>
      <td>4</td>
      <td>325</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
    </tr>
    <tr>
      <th>2585</th>
      <td>2014</td>
      <td>11</td>
      <td>48</td>
      <td>24</td>
      <td>0</td>
      <td>328</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
    </tr>
    <tr>
      <th>2586</th>
      <td>2014</td>
      <td>11</td>
      <td>48</td>
      <td>25</td>
      <td>1</td>
      <td>329</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
    </tr>
    <tr>
      <th>2587</th>
      <td>2014</td>
      <td>11</td>
      <td>48</td>
      <td>26</td>
      <td>2</td>
      <td>330</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
    </tr>
    <tr>
      <th>2588</th>
      <td>2014</td>
      <td>11</td>
      <td>48</td>
      <td>28</td>
      <td>4</td>
      <td>332</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
    </tr>
    <tr>
      <th>2589</th>
      <td>2014</td>
      <td>12</td>
      <td>49</td>
      <td>1</td>
      <td>0</td>
      <td>335</td>
      <td>False</td>
      <td>True</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
    </tr>
    <tr>
      <th>2590</th>
      <td>2014</td>
      <td>12</td>
      <td>49</td>
      <td>2</td>
      <td>1</td>
      <td>336</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
    </tr>
    <tr>
      <th>2591</th>
      <td>2014</td>
      <td>12</td>
      <td>49</td>
      <td>3</td>
      <td>2</td>
      <td>337</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
    </tr>
    <tr>
      <th>2592</th>
      <td>2014</td>
      <td>12</td>
      <td>49</td>
      <td>4</td>
      <td>3</td>
      <td>338</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
    </tr>
    <tr>
      <th>2593</th>
      <td>2014</td>
      <td>12</td>
      <td>49</td>
      <td>5</td>
      <td>4</td>
      <td>339</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
    </tr>
    <tr>
      <th>2594</th>
      <td>2014</td>
      <td>12</td>
      <td>50</td>
      <td>8</td>
      <td>0</td>
      <td>342</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
    </tr>
    <tr>
      <th>2595</th>
      <td>2014</td>
      <td>12</td>
      <td>50</td>
      <td>9</td>
      <td>1</td>
      <td>343</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
    </tr>
    <tr>
      <th>2596</th>
      <td>2014</td>
      <td>12</td>
      <td>50</td>
      <td>10</td>
      <td>2</td>
      <td>344</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
    </tr>
    <tr>
      <th>2597</th>
      <td>2014</td>
      <td>12</td>
      <td>50</td>
      <td>11</td>
      <td>3</td>
      <td>345</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
    </tr>
    <tr>
      <th>2598</th>
      <td>2014</td>
      <td>12</td>
      <td>50</td>
      <td>12</td>
      <td>4</td>
      <td>346</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
    </tr>
    <tr>
      <th>2599</th>
      <td>2014</td>
      <td>12</td>
      <td>51</td>
      <td>15</td>
      <td>0</td>
      <td>349</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
    </tr>
  </tbody>
</table>
<p>2600 rows × 12 columns</p>
</div>




```
# Implementing linear regression
from sklearn.linear_model import LinearRegression
model = LinearRegression()
model.fit(x_train, y_train)
```




    LinearRegression(copy_X=True, fit_intercept=True, n_jobs=None, normalize=False)




```
# Predictions 
preds = model.predict(x_test)
rms = np.sqrt(np.mean(np.power((np.array(y_test)-np.array(preds)),2)))

print(rms)
```

    292.21562094558186



```
# Plot

test['Predictions'] = 0
test['Predictions'] = preds


plt.plot(train['Adj. Close'])
plt.plot(test[['Adj. Close', 'Predictions']])
```

    /usr/local/lib/python3.6/dist-packages/ipykernel_launcher.py:2: SettingWithCopyWarning:
    
    
    A value is trying to be set on a copy of a slice from a DataFrame.
    Try using .loc[row_indexer,col_indexer] = value instead
    
    See the caveats in the documentation: http://pandas.pydata.org/pandas-docs/stable/indexing.html#indexing-view-versus-copy
    
    /usr/local/lib/python3.6/dist-packages/ipykernel_launcher.py:3: SettingWithCopyWarning:
    
    
    A value is trying to be set on a copy of a slice from a DataFrame.
    Try using .loc[row_indexer,col_indexer] = value instead
    
    See the caveats in the documentation: http://pandas.pydata.org/pandas-docs/stable/indexing.html#indexing-view-versus-copy
    





    [<matplotlib.lines.Line2D at 0x7f6dee6f80f0>,
     <matplotlib.lines.Line2D at 0x7f6dee6f82e8>]




![png](images/Stock%20Prediction%20Project_67_2.png)


# k-Nearest Neighbours


```
from sklearn import neighbors
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler(feature_range=(0, 1))
```


```
# scaling the data

x_train_scaled = scaler.fit_transform(x_train)
x_train = pd.DataFrame(x_train_scaled)
x_test_scaled = scaler.fit_transform(x_test)
x_test = pd.DataFrame(x_test_scaled)

```


```
# using gridsearch to find the best value of k

params = {'n_neighbors': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]}
knn = neighbors.KNeighborsRegressor()
model = GridSearchCV(knn, params, cv=5)
```


```
# fitting the model and predicting
model.fit(x_train, y_train)
preds = model.predict(x_test)
```


```
# Results

rms = np.sqrt(np.mean(np.power((np.array(y_test)-np.array(preds)),2)))

print(rms)
```

    589.3454139574452



```
test['Predictions'] = 0
test['Predictions'] = new_preds


plt.plot(train['Adj. Close'])
plt.plot(test[['Adj. Close', 'Predictions']])
```

    /usr/local/lib/python3.6/dist-packages/ipykernel_launcher.py:1: SettingWithCopyWarning:
    
    
    A value is trying to be set on a copy of a slice from a DataFrame.
    Try using .loc[row_indexer,col_indexer] = value instead
    
    See the caveats in the documentation: http://pandas.pydata.org/pandas-docs/stable/indexing.html#indexing-view-versus-copy
    
    /usr/local/lib/python3.6/dist-packages/ipykernel_launcher.py:2: SettingWithCopyWarning:
    
    
    A value is trying to be set on a copy of a slice from a DataFrame.
    Try using .loc[row_indexer,col_indexer] = value instead
    
    See the caveats in the documentation: http://pandas.pydata.org/pandas-docs/stable/indexing.html#indexing-view-versus-copy
    





    [<matplotlib.lines.Line2D at 0x7f6dd249ab70>,
     <matplotlib.lines.Line2D at 0x7f6dd24368d0>]




![png](images/Stock%20Prediction%20Project_74_2.png)


# Multilayer Perceptron


```
import tensorflow as tf
from tensorflow.keras import layers
```


```
model = tf.keras.models.Sequential()
```


```
model.add(tf.keras.layers.Dense(100, activation=tf.nn.relu))
```


```
model.add(tf.keras.layers.Dense(100, activation=tf.nn.relu))

```


```
model.add(tf.keras.layers.Dense(1, activation=tf.nn.relu))
```


```
model.compile(optimizer='adam', loss='mean_squared_error')
```


```
X_train = np.array(x_train)
Y_train = np.array(y_train)
```


```
X_train
```




    array([[0.     , 0.63636, 0.63462, ..., 0.     , 0.     , 0.     ],
           [0.     , 0.63636, 0.63462, ..., 0.     , 0.     , 0.     ],
           [0.     , 0.63636, 0.65385, ..., 0.     , 0.     , 0.     ],
           ...,
           [1.     , 0.72727, 0.75   , ..., 0.     , 0.     , 0.     ],
           [1.     , 0.72727, 0.75   , ..., 0.     , 0.     , 0.     ],
           [1.     , 0.81818, 0.75   , ..., 1.     , 0.     , 0.     ]])




```
X_train.shape
```




    (2800, 12)




```
model.fit(X_train, Y_train, epochs=500)
```

    Epoch 1/500
    2800/2800 [==============================] - 1s 297us/sample - loss: 107042.6812
    Epoch 2/500
    2800/2800 [==============================] - 0s 67us/sample - loss: 41312.2483
    Epoch 3/500
    2800/2800 [==============================] - 0s 60us/sample - loss: 18302.2589
    Epoch 4/500
    2800/2800 [==============================] - 0s 66us/sample - loss: 14739.7240
    Epoch 5/500
    2800/2800 [==============================] - 0s 65us/sample - loss: 11421.5729
    Epoch 6/500
    2800/2800 [==============================] - 0s 64us/sample - loss: 8606.6912
    Epoch 7/500
    2800/2800 [==============================] - 0s 65us/sample - loss: 6625.6495
    Epoch 8/500
    2800/2800 [==============================] - 0s 62us/sample - loss: 5474.2571
    Epoch 9/500
    2800/2800 [==============================] - 0s 60us/sample - loss: 4892.3979
    Epoch 10/500
    2800/2800 [==============================] - 0s 59us/sample - loss: 4633.2238
    Epoch 11/500
    2800/2800 [==============================] - 0s 62us/sample - loss: 4482.4471
    Epoch 12/500
    2800/2800 [==============================] - 0s 59us/sample - loss: 4380.0899
    Epoch 13/500
    2800/2800 [==============================] - 0s 55us/sample - loss: 4300.8690
    Epoch 14/500
    2800/2800 [==============================] - 0s 62us/sample - loss: 4214.3174
    Epoch 15/500
    2800/2800 [==============================] - 0s 59us/sample - loss: 4159.0237
    Epoch 16/500
    2800/2800 [==============================] - 0s 64us/sample - loss: 4102.6403
    Epoch 17/500
    2800/2800 [==============================] - 0s 62us/sample - loss: 4074.3091
    Epoch 18/500
    2800/2800 [==============================] - 0s 62us/sample - loss: 4040.7758
    Epoch 19/500
    2800/2800 [==============================] - 0s 56us/sample - loss: 4000.0518
    Epoch 20/500
    2800/2800 [==============================] - 0s 56us/sample - loss: 4004.0519
    Epoch 21/500
    2800/2800 [==============================] - 0s 60us/sample - loss: 3929.9223
    Epoch 22/500
    2800/2800 [==============================] - 0s 56us/sample - loss: 3899.7539
    Epoch 23/500
    2800/2800 [==============================] - 0s 60us/sample - loss: 3859.3802
    Epoch 24/500
    2800/2800 [==============================] - 0s 54us/sample - loss: 3802.6503
    Epoch 25/500
    2800/2800 [==============================] - 0s 57us/sample - loss: 3777.4047
    Epoch 26/500
    2800/2800 [==============================] - 0s 60us/sample - loss: 3721.2936
    Epoch 27/500
    2800/2800 [==============================] - 0s 59us/sample - loss: 3658.5611
    Epoch 28/500
    2800/2800 [==============================] - 0s 57us/sample - loss: 3596.7238
    Epoch 29/500
    2800/2800 [==============================] - 0s 63us/sample - loss: 3569.7354
    Epoch 30/500
    2800/2800 [==============================] - 0s 57us/sample - loss: 3508.0627
    Epoch 31/500
    2800/2800 [==============================] - 0s 58us/sample - loss: 3432.4962
    Epoch 32/500
    2800/2800 [==============================] - 0s 59us/sample - loss: 3371.8384
    Epoch 33/500
    2800/2800 [==============================] - 0s 55us/sample - loss: 3327.3058
    Epoch 34/500
    2800/2800 [==============================] - 0s 59us/sample - loss: 3239.4707
    Epoch 35/500
    2800/2800 [==============================] - 0s 60us/sample - loss: 3211.6994
    Epoch 36/500
    2800/2800 [==============================] - 0s 55us/sample - loss: 3118.9492
    Epoch 37/500
    2800/2800 [==============================] - 0s 60us/sample - loss: 3059.0352
    Epoch 38/500
    2800/2800 [==============================] - 0s 54us/sample - loss: 2997.7588
    Epoch 39/500
    2800/2800 [==============================] - 0s 57us/sample - loss: 2929.7851
    Epoch 40/500
    2800/2800 [==============================] - 0s 62us/sample - loss: 2875.7662
    Epoch 41/500
    2800/2800 [==============================] - 0s 60us/sample - loss: 2815.8341
    Epoch 42/500
    2800/2800 [==============================] - 0s 55us/sample - loss: 2755.2486
    Epoch 43/500
    2800/2800 [==============================] - 0s 62us/sample - loss: 2743.1073
    Epoch 44/500
    2800/2800 [==============================] - 0s 55us/sample - loss: 2650.5037
    Epoch 45/500
    2800/2800 [==============================] - 0s 62us/sample - loss: 2611.3546
    Epoch 46/500
    2800/2800 [==============================] - 0s 59us/sample - loss: 2559.4127
    Epoch 47/500
    2800/2800 [==============================] - 0s 62us/sample - loss: 2502.4777
    Epoch 48/500
    2800/2800 [==============================] - 0s 59us/sample - loss: 2473.7575
    Epoch 49/500
    2800/2800 [==============================] - 0s 60us/sample - loss: 2427.9666
    Epoch 50/500
    2800/2800 [==============================] - 0s 55us/sample - loss: 2375.8519
    Epoch 51/500
    2800/2800 [==============================] - 0s 59us/sample - loss: 2335.6260
    Epoch 52/500
    2800/2800 [==============================] - 0s 60us/sample - loss: 2302.3150
    Epoch 53/500
    2800/2800 [==============================] - 0s 63us/sample - loss: 2261.2262
    Epoch 54/500
    2800/2800 [==============================] - 0s 57us/sample - loss: 2276.2005
    Epoch 55/500
    2800/2800 [==============================] - 0s 61us/sample - loss: 2217.6836
    Epoch 56/500
    2800/2800 [==============================] - 0s 53us/sample - loss: 2200.5515
    Epoch 57/500
    2800/2800 [==============================] - 0s 69us/sample - loss: 2166.6168
    Epoch 58/500
    2800/2800 [==============================] - 0s 62us/sample - loss: 2149.8538
    Epoch 59/500
    2800/2800 [==============================] - 0s 62us/sample - loss: 2131.2684
    Epoch 60/500
    2800/2800 [==============================] - 0s 65us/sample - loss: 2117.5617
    Epoch 61/500
    2800/2800 [==============================] - 0s 59us/sample - loss: 2136.8635
    Epoch 62/500
    2800/2800 [==============================] - 0s 59us/sample - loss: 2086.9172
    Epoch 63/500
    2800/2800 [==============================] - 0s 60us/sample - loss: 2074.4112
    Epoch 64/500
    2800/2800 [==============================] - 0s 63us/sample - loss: 2070.8835
    Epoch 65/500
    2800/2800 [==============================] - 0s 65us/sample - loss: 2049.8117
    Epoch 66/500
    2800/2800 [==============================] - 0s 56us/sample - loss: 2059.9661
    Epoch 67/500
    2800/2800 [==============================] - 0s 64us/sample - loss: 2042.7345
    Epoch 68/500
    2800/2800 [==============================] - 0s 58us/sample - loss: 2043.8399
    Epoch 69/500
    2800/2800 [==============================] - 0s 64us/sample - loss: 2043.6758
    Epoch 70/500
    2800/2800 [==============================] - 0s 65us/sample - loss: 2010.1117
    Epoch 71/500
    2800/2800 [==============================] - 0s 57us/sample - loss: 2006.6912
    Epoch 72/500
    2800/2800 [==============================] - 0s 60us/sample - loss: 1998.4531
    Epoch 73/500
    2800/2800 [==============================] - 0s 60us/sample - loss: 1984.5140
    Epoch 74/500
    2800/2800 [==============================] - 0s 58us/sample - loss: 1996.5648
    Epoch 75/500
    2800/2800 [==============================] - 0s 60us/sample - loss: 1962.1523
    Epoch 76/500
    2800/2800 [==============================] - 0s 62us/sample - loss: 1955.9846
    Epoch 77/500
    2800/2800 [==============================] - 0s 60us/sample - loss: 1955.2565
    Epoch 78/500
    2800/2800 [==============================] - 0s 61us/sample - loss: 1947.7191
    Epoch 79/500
    2800/2800 [==============================] - 0s 56us/sample - loss: 1953.3301
    Epoch 80/500
    2800/2800 [==============================] - 0s 61us/sample - loss: 1934.9385
    Epoch 81/500
    2800/2800 [==============================] - 0s 60us/sample - loss: 1924.2109
    Epoch 82/500
    2800/2800 [==============================] - 0s 62us/sample - loss: 1910.7859
    Epoch 83/500
    2800/2800 [==============================] - 0s 60us/sample - loss: 1897.3683
    Epoch 84/500
    2800/2800 [==============================] - 0s 59us/sample - loss: 1907.4632
    Epoch 85/500
    2800/2800 [==============================] - 0s 58us/sample - loss: 1891.1020
    Epoch 86/500
    2800/2800 [==============================] - 0s 55us/sample - loss: 1891.9109
    Epoch 87/500
    2800/2800 [==============================] - 0s 60us/sample - loss: 1864.7346
    Epoch 88/500
    2800/2800 [==============================] - 0s 61us/sample - loss: 1862.2231
    Epoch 89/500
    2800/2800 [==============================] - 0s 56us/sample - loss: 1844.3185
    Epoch 90/500
    2800/2800 [==============================] - 0s 59us/sample - loss: 1834.2257
    Epoch 91/500
    2800/2800 [==============================] - 0s 59us/sample - loss: 1833.4358
    Epoch 92/500
    2800/2800 [==============================] - 0s 55us/sample - loss: 1814.5655
    Epoch 93/500
    2800/2800 [==============================] - 0s 58us/sample - loss: 1791.7842
    Epoch 94/500
    2800/2800 [==============================] - 0s 61us/sample - loss: 1780.9216
    Epoch 95/500
    2800/2800 [==============================] - 0s 59us/sample - loss: 1775.3002
    Epoch 96/500
    2800/2800 [==============================] - 0s 58us/sample - loss: 1775.6334
    Epoch 97/500
    2800/2800 [==============================] - 0s 59us/sample - loss: 1741.6012
    Epoch 98/500
    2800/2800 [==============================] - 0s 56us/sample - loss: 1748.9729
    Epoch 99/500
    2800/2800 [==============================] - 0s 61us/sample - loss: 1729.0204
    Epoch 100/500
    2800/2800 [==============================] - 0s 64us/sample - loss: 1717.6277
    Epoch 101/500
    2800/2800 [==============================] - 0s 60us/sample - loss: 1718.6550
    Epoch 102/500
    2800/2800 [==============================] - 0s 55us/sample - loss: 1688.5626
    Epoch 103/500
    2800/2800 [==============================] - 0s 57us/sample - loss: 1678.3398
    Epoch 104/500
    2800/2800 [==============================] - 0s 57us/sample - loss: 1653.9336
    Epoch 105/500
    2800/2800 [==============================] - 0s 60us/sample - loss: 1639.7355
    Epoch 106/500
    2800/2800 [==============================] - 0s 60us/sample - loss: 1652.1106
    Epoch 107/500
    2800/2800 [==============================] - 0s 61us/sample - loss: 1614.5160
    Epoch 108/500
    2800/2800 [==============================] - 0s 60us/sample - loss: 1596.8088
    Epoch 109/500
    2800/2800 [==============================] - 0s 61us/sample - loss: 1614.2061
    Epoch 110/500
    2800/2800 [==============================] - 0s 61us/sample - loss: 1573.5171
    Epoch 111/500
    2800/2800 [==============================] - 0s 61us/sample - loss: 1572.3198
    Epoch 112/500
    2800/2800 [==============================] - 0s 61us/sample - loss: 1559.2580
    Epoch 113/500
    2800/2800 [==============================] - 0s 60us/sample - loss: 1563.9145
    Epoch 114/500
    2800/2800 [==============================] - 0s 56us/sample - loss: 1538.0331
    Epoch 115/500
    2800/2800 [==============================] - 0s 62us/sample - loss: 1522.4755
    Epoch 116/500
    2800/2800 [==============================] - 0s 66us/sample - loss: 1514.9207
    Epoch 117/500
    2800/2800 [==============================] - 0s 72us/sample - loss: 1505.6984
    Epoch 118/500
    2800/2800 [==============================] - 0s 65us/sample - loss: 1497.6740
    Epoch 119/500
    2800/2800 [==============================] - 0s 67us/sample - loss: 1489.2281
    Epoch 120/500
    2800/2800 [==============================] - 0s 63us/sample - loss: 1472.0496
    Epoch 121/500
    2800/2800 [==============================] - 0s 66us/sample - loss: 1455.5400
    Epoch 122/500
    2800/2800 [==============================] - 0s 68us/sample - loss: 1455.3277
    Epoch 123/500
    2800/2800 [==============================] - 0s 72us/sample - loss: 1448.1577
    Epoch 124/500
    2800/2800 [==============================] - 0s 66us/sample - loss: 1442.3502
    Epoch 125/500
    2800/2800 [==============================] - 0s 66us/sample - loss: 1438.5963
    Epoch 126/500
    2800/2800 [==============================] - 0s 65us/sample - loss: 1420.4090
    Epoch 127/500
    2800/2800 [==============================] - 0s 65us/sample - loss: 1410.9522
    Epoch 128/500
    2800/2800 [==============================] - 0s 66us/sample - loss: 1404.9106
    Epoch 129/500
    2800/2800 [==============================] - 0s 64us/sample - loss: 1394.6398
    Epoch 130/500
    2800/2800 [==============================] - 0s 65us/sample - loss: 1394.5640
    Epoch 131/500
    2800/2800 [==============================] - 0s 66us/sample - loss: 1380.8252
    Epoch 132/500
    2800/2800 [==============================] - 0s 68us/sample - loss: 1394.0857
    Epoch 133/500
    2800/2800 [==============================] - 0s 70us/sample - loss: 1379.1661
    Epoch 134/500
    2800/2800 [==============================] - 0s 68us/sample - loss: 1367.0957
    Epoch 135/500
    2800/2800 [==============================] - 0s 67us/sample - loss: 1369.8024
    Epoch 136/500
    2800/2800 [==============================] - 0s 66us/sample - loss: 1372.2572
    Epoch 137/500
    2800/2800 [==============================] - 0s 64us/sample - loss: 1349.9875
    Epoch 138/500
    2800/2800 [==============================] - 0s 66us/sample - loss: 1328.4315
    Epoch 139/500
    2800/2800 [==============================] - 0s 65us/sample - loss: 1345.7265
    Epoch 140/500
    2800/2800 [==============================] - 0s 66us/sample - loss: 1333.4088
    Epoch 141/500
    2800/2800 [==============================] - 0s 65us/sample - loss: 1332.9744
    Epoch 142/500
    2800/2800 [==============================] - 0s 65us/sample - loss: 1316.0742
    Epoch 143/500
    2800/2800 [==============================] - 0s 67us/sample - loss: 1314.9390
    Epoch 144/500
    2800/2800 [==============================] - 0s 69us/sample - loss: 1309.4800
    Epoch 145/500
    2800/2800 [==============================] - 0s 66us/sample - loss: 1307.4666
    Epoch 146/500
    2800/2800 [==============================] - 0s 64us/sample - loss: 1304.8010
    Epoch 147/500
    2800/2800 [==============================] - 0s 60us/sample - loss: 1304.5787
    Epoch 148/500
    2800/2800 [==============================] - 0s 61us/sample - loss: 1306.1439
    Epoch 149/500
    2800/2800 [==============================] - 0s 59us/sample - loss: 1297.4934
    Epoch 150/500
    2800/2800 [==============================] - 0s 60us/sample - loss: 1291.2073
    Epoch 151/500
    2800/2800 [==============================] - 0s 58us/sample - loss: 1302.2408
    Epoch 152/500
    2800/2800 [==============================] - 0s 60us/sample - loss: 1286.4586
    Epoch 153/500
    2800/2800 [==============================] - 0s 59us/sample - loss: 1289.1976
    Epoch 154/500
    2800/2800 [==============================] - 0s 60us/sample - loss: 1290.9848
    Epoch 155/500
    2800/2800 [==============================] - 0s 60us/sample - loss: 1285.6917
    Epoch 156/500
    2800/2800 [==============================] - 0s 62us/sample - loss: 1284.8793
    Epoch 157/500
    2800/2800 [==============================] - 0s 60us/sample - loss: 1276.9912
    Epoch 158/500
    2800/2800 [==============================] - 0s 57us/sample - loss: 1282.7271
    Epoch 159/500
    2800/2800 [==============================] - 0s 59us/sample - loss: 1270.7725
    Epoch 160/500
    2800/2800 [==============================] - 0s 56us/sample - loss: 1275.8668
    Epoch 161/500
    2800/2800 [==============================] - 0s 60us/sample - loss: 1269.0021
    Epoch 162/500
    2800/2800 [==============================] - 0s 59us/sample - loss: 1269.8503
    Epoch 163/500
    2800/2800 [==============================] - 0s 64us/sample - loss: 1272.6825
    Epoch 164/500
    2800/2800 [==============================] - 0s 61us/sample - loss: 1256.9528
    Epoch 165/500
    2800/2800 [==============================] - 0s 61us/sample - loss: 1265.8719
    Epoch 166/500
    2800/2800 [==============================] - 0s 63us/sample - loss: 1274.9810
    Epoch 167/500
    2800/2800 [==============================] - 0s 66us/sample - loss: 1251.9277
    Epoch 168/500
    2800/2800 [==============================] - 0s 63us/sample - loss: 1254.3053
    Epoch 169/500
    2800/2800 [==============================] - 0s 61us/sample - loss: 1250.6730
    Epoch 170/500
    2800/2800 [==============================] - 0s 60us/sample - loss: 1247.9147
    Epoch 171/500
    2800/2800 [==============================] - 0s 60us/sample - loss: 1255.3040
    Epoch 172/500
    2800/2800 [==============================] - 0s 61us/sample - loss: 1246.4044
    Epoch 173/500
    2800/2800 [==============================] - 0s 60us/sample - loss: 1248.9053
    Epoch 174/500
    2800/2800 [==============================] - 0s 66us/sample - loss: 1235.4831
    Epoch 175/500
    2800/2800 [==============================] - 0s 61us/sample - loss: 1235.7729
    Epoch 176/500
    2800/2800 [==============================] - 0s 63us/sample - loss: 1231.1521
    Epoch 177/500
    2800/2800 [==============================] - 0s 62us/sample - loss: 1229.4448
    Epoch 178/500
    2800/2800 [==============================] - 0s 63us/sample - loss: 1235.0874
    Epoch 179/500
    2800/2800 [==============================] - 0s 63us/sample - loss: 1236.4246
    Epoch 180/500
    2800/2800 [==============================] - 0s 62us/sample - loss: 1234.3112
    Epoch 181/500
    2800/2800 [==============================] - 0s 60us/sample - loss: 1241.1206
    Epoch 182/500
    2800/2800 [==============================] - 0s 55us/sample - loss: 1228.0536
    Epoch 183/500
    2800/2800 [==============================] - 0s 61us/sample - loss: 1234.6207
    Epoch 184/500
    2800/2800 [==============================] - 0s 61us/sample - loss: 1218.8249
    Epoch 185/500
    2800/2800 [==============================] - 0s 55us/sample - loss: 1227.5636
    Epoch 186/500
    2800/2800 [==============================] - 0s 61us/sample - loss: 1231.7618
    Epoch 187/500
    2800/2800 [==============================] - 0s 57us/sample - loss: 1211.7460
    Epoch 188/500
    2800/2800 [==============================] - 0s 59us/sample - loss: 1214.7899
    Epoch 189/500
    2800/2800 [==============================] - 0s 64us/sample - loss: 1227.6510
    Epoch 190/500
    2800/2800 [==============================] - 0s 60us/sample - loss: 1221.4233
    Epoch 191/500
    2800/2800 [==============================] - 0s 60us/sample - loss: 1208.9139
    Epoch 192/500
    2800/2800 [==============================] - 0s 60us/sample - loss: 1211.7745
    Epoch 193/500
    2800/2800 [==============================] - 0s 59us/sample - loss: 1212.7552
    Epoch 194/500
    2800/2800 [==============================] - 0s 55us/sample - loss: 1225.5838
    Epoch 195/500
    2800/2800 [==============================] - 0s 56us/sample - loss: 1203.9864
    Epoch 196/500
    2800/2800 [==============================] - 0s 55us/sample - loss: 1192.8778
    Epoch 197/500
    2800/2800 [==============================] - 0s 60us/sample - loss: 1196.5854
    Epoch 198/500
    2800/2800 [==============================] - 0s 59us/sample - loss: 1195.8245
    Epoch 199/500
    2800/2800 [==============================] - 0s 68us/sample - loss: 1205.7381
    Epoch 200/500
    2800/2800 [==============================] - 0s 58us/sample - loss: 1232.1300
    Epoch 201/500
    2800/2800 [==============================] - 0s 56us/sample - loss: 1206.4220
    Epoch 202/500
    2800/2800 [==============================] - 0s 60us/sample - loss: 1208.0591
    Epoch 203/500
    2800/2800 [==============================] - 0s 60us/sample - loss: 1184.6516
    Epoch 204/500
    2800/2800 [==============================] - 0s 59us/sample - loss: 1182.5998
    Epoch 205/500
    2800/2800 [==============================] - 0s 56us/sample - loss: 1181.9274
    Epoch 206/500
    2800/2800 [==============================] - 0s 61us/sample - loss: 1192.6584
    Epoch 207/500
    2800/2800 [==============================] - 0s 61us/sample - loss: 1193.7265
    Epoch 208/500
    2800/2800 [==============================] - 0s 60us/sample - loss: 1186.7215
    Epoch 209/500
    2800/2800 [==============================] - 0s 59us/sample - loss: 1199.1962
    Epoch 210/500
    2800/2800 [==============================] - 0s 58us/sample - loss: 1173.3175
    Epoch 211/500
    2800/2800 [==============================] - 0s 60us/sample - loss: 1170.1261
    Epoch 212/500
    2800/2800 [==============================] - 0s 60us/sample - loss: 1168.3255
    Epoch 213/500
    2800/2800 [==============================] - 0s 61us/sample - loss: 1168.4484
    Epoch 214/500
    2800/2800 [==============================] - 0s 60us/sample - loss: 1177.4722
    Epoch 215/500
    2800/2800 [==============================] - 0s 62us/sample - loss: 1167.2033
    Epoch 216/500
    2800/2800 [==============================] - 0s 55us/sample - loss: 1168.7120
    Epoch 217/500
    2800/2800 [==============================] - 0s 59us/sample - loss: 1161.4171
    Epoch 218/500
    2800/2800 [==============================] - 0s 56us/sample - loss: 1179.3264
    Epoch 219/500
    2800/2800 [==============================] - 0s 60us/sample - loss: 1168.1229
    Epoch 220/500
    2800/2800 [==============================] - 0s 58us/sample - loss: 1163.9322
    Epoch 221/500
    2800/2800 [==============================] - 0s 57us/sample - loss: 1168.4568
    Epoch 222/500
    2800/2800 [==============================] - 0s 61us/sample - loss: 1155.7791
    Epoch 223/500
    2800/2800 [==============================] - 0s 59us/sample - loss: 1159.8121
    Epoch 224/500
    2800/2800 [==============================] - 0s 60us/sample - loss: 1144.4758
    Epoch 225/500
    2800/2800 [==============================] - 0s 65us/sample - loss: 1138.5819
    Epoch 226/500
    2800/2800 [==============================] - 0s 68us/sample - loss: 1150.9946
    Epoch 227/500
    2800/2800 [==============================] - 0s 55us/sample - loss: 1142.0983
    Epoch 228/500
    2800/2800 [==============================] - 0s 62us/sample - loss: 1139.8449
    Epoch 229/500
    2800/2800 [==============================] - 0s 56us/sample - loss: 1138.2159
    Epoch 230/500
    2800/2800 [==============================] - 0s 58us/sample - loss: 1132.8268
    Epoch 231/500
    2800/2800 [==============================] - 0s 63us/sample - loss: 1144.6047
    Epoch 232/500
    2800/2800 [==============================] - 0s 60us/sample - loss: 1145.6636
    Epoch 233/500
    2800/2800 [==============================] - 0s 56us/sample - loss: 1131.2519
    Epoch 234/500
    2800/2800 [==============================] - 0s 67us/sample - loss: 1163.3628
    Epoch 235/500
    2800/2800 [==============================] - 0s 55us/sample - loss: 1132.9240
    Epoch 236/500
    2800/2800 [==============================] - 0s 61us/sample - loss: 1123.2252
    Epoch 237/500
    2800/2800 [==============================] - 0s 62us/sample - loss: 1143.7624
    Epoch 238/500
    2800/2800 [==============================] - 0s 66us/sample - loss: 1131.8123
    Epoch 239/500
    2800/2800 [==============================] - 0s 55us/sample - loss: 1120.6352
    Epoch 240/500
    2800/2800 [==============================] - 0s 61us/sample - loss: 1123.3857
    Epoch 241/500
    2800/2800 [==============================] - 0s 66us/sample - loss: 1129.1544
    Epoch 242/500
    2800/2800 [==============================] - 0s 62us/sample - loss: 1121.9190
    Epoch 243/500
    2800/2800 [==============================] - 0s 62us/sample - loss: 1116.0550
    Epoch 244/500
    2800/2800 [==============================] - 0s 59us/sample - loss: 1126.5603
    Epoch 245/500
    2800/2800 [==============================] - 0s 59us/sample - loss: 1115.0059
    Epoch 246/500
    2800/2800 [==============================] - 0s 60us/sample - loss: 1123.4432
    Epoch 247/500
    2800/2800 [==============================] - 0s 58us/sample - loss: 1108.1146
    Epoch 248/500
    2800/2800 [==============================] - 0s 62us/sample - loss: 1109.9104
    Epoch 249/500
    2800/2800 [==============================] - 0s 56us/sample - loss: 1115.1202
    Epoch 250/500
    2800/2800 [==============================] - 0s 60us/sample - loss: 1116.5792
    Epoch 251/500
    2800/2800 [==============================] - 0s 60us/sample - loss: 1109.9456
    Epoch 252/500
    2800/2800 [==============================] - 0s 60us/sample - loss: 1111.4490
    Epoch 253/500
    2800/2800 [==============================] - 0s 60us/sample - loss: 1108.3779
    Epoch 254/500
    2800/2800 [==============================] - 0s 61us/sample - loss: 1104.7345
    Epoch 255/500
    2800/2800 [==============================] - 0s 59us/sample - loss: 1112.3371
    Epoch 256/500
    2800/2800 [==============================] - 0s 64us/sample - loss: 1104.8332
    Epoch 257/500
    2800/2800 [==============================] - 0s 61us/sample - loss: 1114.8357
    Epoch 258/500
    2800/2800 [==============================] - 0s 61us/sample - loss: 1101.6882
    Epoch 259/500
    2800/2800 [==============================] - 0s 60us/sample - loss: 1101.5345
    Epoch 260/500
    2800/2800 [==============================] - 0s 56us/sample - loss: 1103.9248
    Epoch 261/500
    2800/2800 [==============================] - 0s 66us/sample - loss: 1097.1669
    Epoch 262/500
    2800/2800 [==============================] - 0s 59us/sample - loss: 1115.2441
    Epoch 263/500
    2800/2800 [==============================] - 0s 60us/sample - loss: 1112.0905
    Epoch 264/500
    2800/2800 [==============================] - 0s 60us/sample - loss: 1125.7278
    Epoch 265/500
    2800/2800 [==============================] - 0s 61us/sample - loss: 1095.7451
    Epoch 266/500
    2800/2800 [==============================] - 0s 60us/sample - loss: 1097.3662
    Epoch 267/500
    2800/2800 [==============================] - 0s 66us/sample - loss: 1099.4183
    Epoch 268/500
    2800/2800 [==============================] - 0s 57us/sample - loss: 1104.0748
    Epoch 269/500
    2800/2800 [==============================] - 0s 63us/sample - loss: 1091.5516
    Epoch 270/500
    2800/2800 [==============================] - 0s 59us/sample - loss: 1111.4074
    Epoch 271/500
    2800/2800 [==============================] - 0s 60us/sample - loss: 1087.1720
    Epoch 272/500
    2800/2800 [==============================] - 0s 58us/sample - loss: 1100.7326
    Epoch 273/500
    2800/2800 [==============================] - 0s 61us/sample - loss: 1100.8644
    Epoch 274/500
    2800/2800 [==============================] - 0s 58us/sample - loss: 1092.4085
    Epoch 275/500
    2800/2800 [==============================] - 0s 58us/sample - loss: 1096.4041
    Epoch 276/500
    2800/2800 [==============================] - 0s 60us/sample - loss: 1092.8723
    Epoch 277/500
    2800/2800 [==============================] - 0s 59us/sample - loss: 1088.8798
    Epoch 278/500
    2800/2800 [==============================] - 0s 60us/sample - loss: 1085.1037
    Epoch 279/500
    2800/2800 [==============================] - 0s 62us/sample - loss: 1081.6819
    Epoch 280/500
    2800/2800 [==============================] - 0s 60us/sample - loss: 1088.1270
    Epoch 281/500
    2800/2800 [==============================] - 0s 55us/sample - loss: 1085.6152
    Epoch 282/500
    2800/2800 [==============================] - 0s 62us/sample - loss: 1087.3699
    Epoch 283/500
    2800/2800 [==============================] - 0s 55us/sample - loss: 1079.3115
    Epoch 284/500
    2800/2800 [==============================] - 0s 63us/sample - loss: 1075.5549
    Epoch 285/500
    2800/2800 [==============================] - 0s 59us/sample - loss: 1075.9227
    Epoch 286/500
    2800/2800 [==============================] - 0s 60us/sample - loss: 1079.0676
    Epoch 287/500
    2800/2800 [==============================] - 0s 60us/sample - loss: 1083.7895
    Epoch 288/500
    2800/2800 [==============================] - 0s 58us/sample - loss: 1080.3468
    Epoch 289/500
    2800/2800 [==============================] - 0s 55us/sample - loss: 1085.7862
    Epoch 290/500
    2800/2800 [==============================] - 0s 61us/sample - loss: 1078.2151
    Epoch 291/500
    2800/2800 [==============================] - 0s 62us/sample - loss: 1068.3046
    Epoch 292/500
    2800/2800 [==============================] - 0s 60us/sample - loss: 1075.6586
    Epoch 293/500
    2800/2800 [==============================] - 0s 61us/sample - loss: 1071.6740
    Epoch 294/500
    2800/2800 [==============================] - 0s 66us/sample - loss: 1072.1976
    Epoch 295/500
    2800/2800 [==============================] - 0s 56us/sample - loss: 1074.4623
    Epoch 296/500
    2800/2800 [==============================] - 0s 61us/sample - loss: 1077.1401
    Epoch 297/500
    2800/2800 [==============================] - 0s 62us/sample - loss: 1083.3203
    Epoch 298/500
    2800/2800 [==============================] - 0s 60us/sample - loss: 1084.7689
    Epoch 299/500
    2800/2800 [==============================] - 0s 59us/sample - loss: 1080.1851
    Epoch 300/500
    2800/2800 [==============================] - 0s 56us/sample - loss: 1077.2795
    Epoch 301/500
    2800/2800 [==============================] - 0s 61us/sample - loss: 1067.7424
    Epoch 302/500
    2800/2800 [==============================] - 0s 56us/sample - loss: 1068.0804
    Epoch 303/500
    2800/2800 [==============================] - 0s 62us/sample - loss: 1066.2189
    Epoch 304/500
    2800/2800 [==============================] - 0s 61us/sample - loss: 1070.1389
    Epoch 305/500
    2800/2800 [==============================] - 0s 55us/sample - loss: 1068.9136
    Epoch 306/500
    2800/2800 [==============================] - 0s 56us/sample - loss: 1079.3364
    Epoch 307/500
    2800/2800 [==============================] - 0s 64us/sample - loss: 1078.0581
    Epoch 308/500
    2800/2800 [==============================] - 0s 56us/sample - loss: 1100.6148
    Epoch 309/500
    2800/2800 [==============================] - 0s 64us/sample - loss: 1065.8257
    Epoch 310/500
    2800/2800 [==============================] - 0s 55us/sample - loss: 1066.7104
    Epoch 311/500
    2800/2800 [==============================] - 0s 62us/sample - loss: 1060.9756
    Epoch 312/500
    2800/2800 [==============================] - 0s 60us/sample - loss: 1064.4785
    Epoch 313/500
    2800/2800 [==============================] - 0s 55us/sample - loss: 1062.8892
    Epoch 314/500
    2800/2800 [==============================] - 0s 59us/sample - loss: 1076.1493
    Epoch 315/500
    2800/2800 [==============================] - 0s 63us/sample - loss: 1075.2968
    Epoch 316/500
    2800/2800 [==============================] - 0s 62us/sample - loss: 1062.3199
    Epoch 317/500
    2800/2800 [==============================] - 0s 57us/sample - loss: 1064.0583
    Epoch 318/500
    2800/2800 [==============================] - 0s 60us/sample - loss: 1073.0384
    Epoch 319/500
    2800/2800 [==============================] - 0s 56us/sample - loss: 1060.9311
    Epoch 320/500
    2800/2800 [==============================] - 0s 61us/sample - loss: 1060.9842
    Epoch 321/500
    2800/2800 [==============================] - 0s 60us/sample - loss: 1065.6132
    Epoch 322/500
    2800/2800 [==============================] - 0s 62us/sample - loss: 1058.8536
    Epoch 323/500
    2800/2800 [==============================] - 0s 62us/sample - loss: 1056.9967
    Epoch 324/500
    2800/2800 [==============================] - 0s 60us/sample - loss: 1051.3154
    Epoch 325/500
    2800/2800 [==============================] - 0s 56us/sample - loss: 1052.8784
    Epoch 326/500
    2800/2800 [==============================] - 0s 60us/sample - loss: 1042.9993
    Epoch 327/500
    2800/2800 [==============================] - 0s 64us/sample - loss: 1044.8463
    Epoch 328/500
    2800/2800 [==============================] - 0s 60us/sample - loss: 1051.4864
    Epoch 329/500
    2800/2800 [==============================] - 0s 61us/sample - loss: 1043.7003
    Epoch 330/500
    2800/2800 [==============================] - 0s 60us/sample - loss: 1051.8080
    Epoch 331/500
    2800/2800 [==============================] - 0s 53us/sample - loss: 1054.5417
    Epoch 332/500
    2800/2800 [==============================] - 0s 62us/sample - loss: 1057.1723
    Epoch 333/500
    2800/2800 [==============================] - 0s 56us/sample - loss: 1053.7270
    Epoch 334/500
    2800/2800 [==============================] - 0s 61us/sample - loss: 1060.2016
    Epoch 335/500
    2800/2800 [==============================] - 0s 60us/sample - loss: 1049.5115
    Epoch 336/500
    2800/2800 [==============================] - 0s 59us/sample - loss: 1041.7271
    Epoch 337/500
    2800/2800 [==============================] - 0s 60us/sample - loss: 1031.7273
    Epoch 338/500
    2800/2800 [==============================] - 0s 56us/sample - loss: 1044.8150
    Epoch 339/500
    2800/2800 [==============================] - 0s 62us/sample - loss: 1046.0760
    Epoch 340/500
    2800/2800 [==============================] - 0s 55us/sample - loss: 1049.0824
    Epoch 341/500
    2800/2800 [==============================] - 0s 61us/sample - loss: 1040.2875
    Epoch 342/500
    2800/2800 [==============================] - 0s 60us/sample - loss: 1030.4299
    Epoch 343/500
    2800/2800 [==============================] - 0s 60us/sample - loss: 1043.7071
    Epoch 344/500
    2800/2800 [==============================] - 0s 61us/sample - loss: 1034.6867
    Epoch 345/500
    2800/2800 [==============================] - 0s 61us/sample - loss: 1037.4047
    Epoch 346/500
    2800/2800 [==============================] - 0s 59us/sample - loss: 1039.7557
    Epoch 347/500
    2800/2800 [==============================] - 0s 56us/sample - loss: 1024.1343
    Epoch 348/500
    2800/2800 [==============================] - 0s 61us/sample - loss: 1027.6918
    Epoch 349/500
    2800/2800 [==============================] - 0s 60us/sample - loss: 1018.4648
    Epoch 350/500
    2800/2800 [==============================] - 0s 62us/sample - loss: 1013.8253
    Epoch 351/500
    2800/2800 [==============================] - 0s 60us/sample - loss: 1013.7689
    Epoch 352/500
    2800/2800 [==============================] - 0s 59us/sample - loss: 1023.0029
    Epoch 353/500
    2800/2800 [==============================] - 0s 59us/sample - loss: 1017.0246
    Epoch 354/500
    2800/2800 [==============================] - 0s 63us/sample - loss: 1024.0550
    Epoch 355/500
    2800/2800 [==============================] - 0s 67us/sample - loss: 1009.9415
    Epoch 356/500
    2800/2800 [==============================] - 0s 62us/sample - loss: 1003.8997
    Epoch 357/500
    2800/2800 [==============================] - 0s 62us/sample - loss: 1006.5315
    Epoch 358/500
    2800/2800 [==============================] - 0s 61us/sample - loss: 1016.5423
    Epoch 359/500
    2800/2800 [==============================] - 0s 63us/sample - loss: 1010.5035
    Epoch 360/500
    2800/2800 [==============================] - 0s 64us/sample - loss: 1002.4500
    Epoch 361/500
    2800/2800 [==============================] - 0s 58us/sample - loss: 1015.2172
    Epoch 362/500
    2800/2800 [==============================] - 0s 63us/sample - loss: 1007.1971
    Epoch 363/500
    2800/2800 [==============================] - 0s 61us/sample - loss: 1006.8336
    Epoch 364/500
    2800/2800 [==============================] - 0s 60us/sample - loss: 1007.2265
    Epoch 365/500
    2800/2800 [==============================] - 0s 59us/sample - loss: 1007.9150
    Epoch 366/500
    2800/2800 [==============================] - 0s 62us/sample - loss: 1006.6157
    Epoch 367/500
    2800/2800 [==============================] - 0s 63us/sample - loss: 997.9993
    Epoch 368/500
    2800/2800 [==============================] - 0s 63us/sample - loss: 1010.8255
    Epoch 369/500
    2800/2800 [==============================] - 0s 60us/sample - loss: 1010.5538
    Epoch 370/500
    2800/2800 [==============================] - 0s 61us/sample - loss: 997.4726
    Epoch 371/500
    2800/2800 [==============================] - 0s 61us/sample - loss: 997.9302
    Epoch 372/500
    2800/2800 [==============================] - 0s 58us/sample - loss: 988.6962
    Epoch 373/500
    2800/2800 [==============================] - 0s 63us/sample - loss: 995.6228
    Epoch 374/500
    2800/2800 [==============================] - 0s 62us/sample - loss: 989.1748
    Epoch 375/500
    2800/2800 [==============================] - 0s 60us/sample - loss: 992.7116
    Epoch 376/500
    2800/2800 [==============================] - 0s 64us/sample - loss: 987.3237
    Epoch 377/500
    2800/2800 [==============================] - 0s 63us/sample - loss: 992.6085
    Epoch 378/500
    2800/2800 [==============================] - 0s 62us/sample - loss: 994.6691
    Epoch 379/500
    2800/2800 [==============================] - 0s 65us/sample - loss: 1008.5694
    Epoch 380/500
    2800/2800 [==============================] - 0s 63us/sample - loss: 986.9961
    Epoch 381/500
    2800/2800 [==============================] - 0s 61us/sample - loss: 985.3143
    Epoch 382/500
    2800/2800 [==============================] - 0s 64us/sample - loss: 980.2779
    Epoch 383/500
    2800/2800 [==============================] - 0s 64us/sample - loss: 983.7935
    Epoch 384/500
    2800/2800 [==============================] - 0s 64us/sample - loss: 982.4315
    Epoch 385/500
    2800/2800 [==============================] - 0s 60us/sample - loss: 978.9523
    Epoch 386/500
    2800/2800 [==============================] - 0s 61us/sample - loss: 972.5764
    Epoch 387/500
    2800/2800 [==============================] - 0s 62us/sample - loss: 974.3637
    Epoch 388/500
    2800/2800 [==============================] - 0s 59us/sample - loss: 984.3422
    Epoch 389/500
    2800/2800 [==============================] - 0s 61us/sample - loss: 980.3249
    Epoch 390/500
    2800/2800 [==============================] - 0s 66us/sample - loss: 976.2236
    Epoch 391/500
    2800/2800 [==============================] - 0s 61us/sample - loss: 981.1634
    Epoch 392/500
    2800/2800 [==============================] - 0s 63us/sample - loss: 969.4153
    Epoch 393/500
    2800/2800 [==============================] - 0s 60us/sample - loss: 981.6107
    Epoch 394/500
    2800/2800 [==============================] - 0s 61us/sample - loss: 968.6300
    Epoch 395/500
    2800/2800 [==============================] - 0s 64us/sample - loss: 979.7458
    Epoch 396/500
    2800/2800 [==============================] - 0s 68us/sample - loss: 962.2366
    Epoch 397/500
    2800/2800 [==============================] - 0s 62us/sample - loss: 981.2674
    Epoch 398/500
    2800/2800 [==============================] - 0s 64us/sample - loss: 978.5301
    Epoch 399/500
    2800/2800 [==============================] - 0s 67us/sample - loss: 969.2405
    Epoch 400/500
    2800/2800 [==============================] - 0s 62us/sample - loss: 961.5628
    Epoch 401/500
    2800/2800 [==============================] - 0s 66us/sample - loss: 970.1200
    Epoch 402/500
    2800/2800 [==============================] - 0s 65us/sample - loss: 977.9712
    Epoch 403/500
    2800/2800 [==============================] - 0s 64us/sample - loss: 962.8104
    Epoch 404/500
    2800/2800 [==============================] - 0s 61us/sample - loss: 969.9754
    Epoch 405/500
    2800/2800 [==============================] - 0s 67us/sample - loss: 968.4667
    Epoch 406/500
    2800/2800 [==============================] - 0s 69us/sample - loss: 956.4312
    Epoch 407/500
    2800/2800 [==============================] - 0s 65us/sample - loss: 967.2131
    Epoch 408/500
    2800/2800 [==============================] - 0s 63us/sample - loss: 972.8434
    Epoch 409/500
    2800/2800 [==============================] - 0s 60us/sample - loss: 970.3013
    Epoch 410/500
    2800/2800 [==============================] - 0s 64us/sample - loss: 956.3091
    Epoch 411/500
    2800/2800 [==============================] - 0s 65us/sample - loss: 955.3719
    Epoch 412/500
    2800/2800 [==============================] - 0s 75us/sample - loss: 968.8319
    Epoch 413/500
    2800/2800 [==============================] - 0s 63us/sample - loss: 964.6352
    Epoch 414/500
    2800/2800 [==============================] - 0s 62us/sample - loss: 956.8947
    Epoch 415/500
    2800/2800 [==============================] - 0s 65us/sample - loss: 969.1176
    Epoch 416/500
    2800/2800 [==============================] - 0s 66us/sample - loss: 955.3204
    Epoch 417/500
    2800/2800 [==============================] - 0s 66us/sample - loss: 947.9882
    Epoch 418/500
    2800/2800 [==============================] - 0s 64us/sample - loss: 963.2110
    Epoch 419/500
    2800/2800 [==============================] - 0s 64us/sample - loss: 956.1462
    Epoch 420/500
    2800/2800 [==============================] - 0s 64us/sample - loss: 967.4477
    Epoch 421/500
    2800/2800 [==============================] - 0s 65us/sample - loss: 958.8824
    Epoch 422/500
    2800/2800 [==============================] - 0s 64us/sample - loss: 955.9722
    Epoch 423/500
    2800/2800 [==============================] - 0s 62us/sample - loss: 943.9016
    Epoch 424/500
    2800/2800 [==============================] - 0s 67us/sample - loss: 946.2346
    Epoch 425/500
    2800/2800 [==============================] - 0s 67us/sample - loss: 957.0402
    Epoch 426/500
    2800/2800 [==============================] - 0s 60us/sample - loss: 947.0486
    Epoch 427/500
    2800/2800 [==============================] - 0s 61us/sample - loss: 946.5983
    Epoch 428/500
    2800/2800 [==============================] - 0s 65us/sample - loss: 953.2834
    Epoch 429/500
    2800/2800 [==============================] - 0s 65us/sample - loss: 945.5103
    Epoch 430/500
    2800/2800 [==============================] - 0s 62us/sample - loss: 964.1069
    Epoch 431/500
    2800/2800 [==============================] - 0s 59us/sample - loss: 949.8702
    Epoch 432/500
    2800/2800 [==============================] - 0s 59us/sample - loss: 946.5142
    Epoch 433/500
    2800/2800 [==============================] - 0s 62us/sample - loss: 965.2561
    Epoch 434/500
    2800/2800 [==============================] - 0s 62us/sample - loss: 952.1427
    Epoch 435/500
    2800/2800 [==============================] - 0s 61us/sample - loss: 954.9595
    Epoch 436/500
    2800/2800 [==============================] - 0s 66us/sample - loss: 937.4476
    Epoch 437/500
    2800/2800 [==============================] - 0s 71us/sample - loss: 945.1045
    Epoch 438/500
    2800/2800 [==============================] - 0s 63us/sample - loss: 939.1078
    Epoch 439/500
    2800/2800 [==============================] - 0s 61us/sample - loss: 939.0538
    Epoch 440/500
    2800/2800 [==============================] - 0s 65us/sample - loss: 947.7939
    Epoch 441/500
    2800/2800 [==============================] - 0s 64us/sample - loss: 942.8379
    Epoch 442/500
    2800/2800 [==============================] - 0s 61us/sample - loss: 943.3696
    Epoch 443/500
    2800/2800 [==============================] - 0s 61us/sample - loss: 942.1774
    Epoch 444/500
    2800/2800 [==============================] - 0s 61us/sample - loss: 939.1387
    Epoch 445/500
    2800/2800 [==============================] - 0s 59us/sample - loss: 948.7494
    Epoch 446/500
    2800/2800 [==============================] - 0s 65us/sample - loss: 943.7548
    Epoch 447/500
    2800/2800 [==============================] - 0s 62us/sample - loss: 932.7881
    Epoch 448/500
    2800/2800 [==============================] - 0s 61us/sample - loss: 939.7122
    Epoch 449/500
    2800/2800 [==============================] - 0s 56us/sample - loss: 932.3667
    Epoch 450/500
    2800/2800 [==============================] - 0s 64us/sample - loss: 960.0179
    Epoch 451/500
    2800/2800 [==============================] - 0s 62us/sample - loss: 940.8460
    Epoch 452/500
    2800/2800 [==============================] - 0s 61us/sample - loss: 949.1144
    Epoch 453/500
    2800/2800 [==============================] - 0s 65us/sample - loss: 936.2326
    Epoch 454/500
    2800/2800 [==============================] - 0s 60us/sample - loss: 930.4036
    Epoch 455/500
    2800/2800 [==============================] - 0s 62us/sample - loss: 938.6048
    Epoch 456/500
    2800/2800 [==============================] - 0s 62us/sample - loss: 937.3702
    Epoch 457/500
    2800/2800 [==============================] - 0s 63us/sample - loss: 938.8040
    Epoch 458/500
    2800/2800 [==============================] - 0s 61us/sample - loss: 959.1447
    Epoch 459/500
    2800/2800 [==============================] - 0s 59us/sample - loss: 929.1206
    Epoch 460/500
    2800/2800 [==============================] - 0s 60us/sample - loss: 934.5046
    Epoch 461/500
    2800/2800 [==============================] - 0s 61us/sample - loss: 937.0909
    Epoch 462/500
    2800/2800 [==============================] - 0s 65us/sample - loss: 932.0309
    Epoch 463/500
    2800/2800 [==============================] - 0s 56us/sample - loss: 946.6648
    Epoch 464/500
    2800/2800 [==============================] - 0s 63us/sample - loss: 926.3534
    Epoch 465/500
    2800/2800 [==============================] - 0s 60us/sample - loss: 926.0744
    Epoch 466/500
    2800/2800 [==============================] - 0s 59us/sample - loss: 928.1909
    Epoch 467/500
    2800/2800 [==============================] - 0s 64us/sample - loss: 921.8013
    Epoch 468/500
    2800/2800 [==============================] - 0s 62us/sample - loss: 942.3816
    Epoch 469/500
    2800/2800 [==============================] - 0s 75us/sample - loss: 928.0069
    Epoch 470/500
    2800/2800 [==============================] - 0s 64us/sample - loss: 923.5392
    Epoch 471/500
    2800/2800 [==============================] - 0s 60us/sample - loss: 929.2638
    Epoch 472/500
    2800/2800 [==============================] - 0s 63us/sample - loss: 933.0489
    Epoch 473/500
    2800/2800 [==============================] - 0s 62us/sample - loss: 926.7229
    Epoch 474/500
    2800/2800 [==============================] - 0s 61us/sample - loss: 923.6627
    Epoch 475/500
    2800/2800 [==============================] - 0s 58us/sample - loss: 924.1287
    Epoch 476/500
    2800/2800 [==============================] - 0s 58us/sample - loss: 935.6834
    Epoch 477/500
    2800/2800 [==============================] - 0s 59us/sample - loss: 937.3834
    Epoch 478/500
    2800/2800 [==============================] - 0s 62us/sample - loss: 934.3209
    Epoch 479/500
    2800/2800 [==============================] - 0s 58us/sample - loss: 919.9716
    Epoch 480/500
    2800/2800 [==============================] - 0s 67us/sample - loss: 943.3614
    Epoch 481/500
    2800/2800 [==============================] - 0s 59us/sample - loss: 923.1147
    Epoch 482/500
    2800/2800 [==============================] - 0s 61us/sample - loss: 921.4427
    Epoch 483/500
    2800/2800 [==============================] - 0s 59us/sample - loss: 913.0294
    Epoch 484/500
    2800/2800 [==============================] - 0s 63us/sample - loss: 915.4022
    Epoch 485/500
    2800/2800 [==============================] - 0s 56us/sample - loss: 922.7092
    Epoch 486/500
    2800/2800 [==============================] - 0s 60us/sample - loss: 932.7826
    Epoch 487/500
    2800/2800 [==============================] - 0s 60us/sample - loss: 919.0049
    Epoch 488/500
    2800/2800 [==============================] - 0s 61us/sample - loss: 913.9195
    Epoch 489/500
    2800/2800 [==============================] - 0s 62us/sample - loss: 928.3425
    Epoch 490/500
    2800/2800 [==============================] - 0s 61us/sample - loss: 939.2139
    Epoch 491/500
    2800/2800 [==============================] - 0s 59us/sample - loss: 929.5971
    Epoch 492/500
    2800/2800 [==============================] - 0s 64us/sample - loss: 912.6969
    Epoch 493/500
    2800/2800 [==============================] - 0s 61us/sample - loss: 912.2640
    Epoch 494/500
    2800/2800 [==============================] - 0s 63us/sample - loss: 930.4466
    Epoch 495/500
    2800/2800 [==============================] - 0s 62us/sample - loss: 920.4040
    Epoch 496/500
    2800/2800 [==============================] - 0s 62us/sample - loss: 914.7307
    Epoch 497/500
    2800/2800 [==============================] - 0s 63us/sample - loss: 913.2255
    Epoch 498/500
    2800/2800 [==============================] - 0s 61us/sample - loss: 924.5638
    Epoch 499/500
    2800/2800 [==============================] - 0s 63us/sample - loss: 906.7920
    Epoch 500/500
    2800/2800 [==============================] - 0s 58us/sample - loss: 916.3150





    <tensorflow.python.keras.callbacks.History at 0x7f6dd240abe0>




```
preds = model.predict(x_test)
```


```
# Results

rms = np.sqrt(np.mean(np.power((np.array(y_test)-np.array(preds)),2)))

print(rms)
```

    600.8828804286148



```
test['Predictions'] = 0
test['Predictions'] = preds


plt.plot(train['Adj. Close'])
plt.plot(test[['Adj. Close', 'Predictions']])
```

    /usr/local/lib/python3.6/dist-packages/ipykernel_launcher.py:1: SettingWithCopyWarning:
    
    
    A value is trying to be set on a copy of a slice from a DataFrame.
    Try using .loc[row_indexer,col_indexer] = value instead
    
    See the caveats in the documentation: http://pandas.pydata.org/pandas-docs/stable/indexing.html#indexing-view-versus-copy
    
    /usr/local/lib/python3.6/dist-packages/ipykernel_launcher.py:2: SettingWithCopyWarning:
    
    
    A value is trying to be set on a copy of a slice from a DataFrame.
    Try using .loc[row_indexer,col_indexer] = value instead
    
    See the caveats in the documentation: http://pandas.pydata.org/pandas-docs/stable/indexing.html#indexing-view-versus-copy
    





    [<matplotlib.lines.Line2D at 0x7f6dd20c25f8>,
     <matplotlib.lines.Line2D at 0x7f6dd20c2828>]




![png](images/Stock%20Prediction%20Project_88_2.png)


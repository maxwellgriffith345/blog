+++
title = 'Linear Regression'
date = 2024-09-13T17:46:11-05:00
+++

Applying concepts from [Forecasting Principles and Practice](https://otexts.com/fpp3/expsmooth.html) to electricty prices in Python

<!--more-->

### Intro

This is part three in my series to apply the time series forecasting methods presented by Hyndmam and Athanasopoulos in python. I use real world generator location marginal prices that were pulled from South West Power Pool. This section covers simple linear regression using sci-kit learn and Stats Models

The features I selected are temperature and wind speed at the generator site. This is more of a toy example, as there I did not fully explore which features could be used to improve the model.

**Concepts**: Linear Regression, Seasonal Dummy Variables, Cross-Validation, Model Selection

**Libraries**: Sci-Kit Learn, Stats Models, Matplotlib, Seaborn, Pandas



```python
import pandas as pd
import numpy as np
import seaborn as sns
%matplotlib inline
import matplotlib.pyplot as plt
from datetime import datetime
```


```python
from sklearn.model_selection import TimeSeriesSplit
from sklearn.model_selection import cross_validate
from sklearn.pipeline import make_pipeline
from sklearn import linear_model
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
```


```python
import statsmodels.api as sm
```


### Set Up the Data

```python
#parse datetime with weird timezone format
def parse_datetime_with_timezone(dt_str):
    datetime_str = dt_str[:-3] + dt_str[-2:]
    return pd.to_datetime(datetime_str, utc=True, format="%Y-%m-%d %H:%M:%S%z")

# TODO RE-Write without using "date_parser"
df = pd.read_csv("LMPSWide2023.csv", parse_dates=['Time'], date_parser=parse_datetime_with_timezone, index_col='Time')

# TODO Make sure this is the right timezone
df.index = df.index.tz_convert('America/New_York')

#Set the frequency
#we need to do this before using in STL
df= df.asfreq(freq='h')
```

```python
df.drop(columns=["KCPLIATANUNIAT2","KCPLLACYGNEUNLAC2","SECI.KCPS.CIMARRON", "WR.LEC.4", "WR.JEC.2", "WR.JEC.3", "SECI.KCPS.CIMARRON"],
        inplace=True)
```




```python
#let's do regression on one price node
cimarron=df[["SECI_CIMARRON"]]

#Get the weather data
weather = pd.read_csv("CimarronWeather23.csv")
cimarron["temp"]=weather["temperature"].values
cimarron["wind"]=weather["wind_speed"].values

#Hold back the last week as the final test set
cimarron_train = cimarron.iloc[:-168]
cimarron_test = cimarron.iloc[-168:]
X=cimarron_train.drop("SECI_CIMARRON", axis="columns")
y=cimarron_train.drop(["temp", "wind"], axis="columns")

#Add constant column for regression
X = sm.add_constant(X)
```


```python
#Quick visualization to see any relation between features and price
fig, (ax1, ax2) = plt.subplots(1,2, figsize=(12,7))
cimarron_train.plot(ax=ax1)
cimarron.loc["2023-11"].plot(ax=ax2)
ax1.set_title("2023 Node Price, Temp, WindSpeed")
ax2.set_title("November Node Price, Temp, WindSpeed")
```

{{< figure src="/Regression_files/Regression_8_1.png"
title="" >}}


### Basic OLS
* Let's try it in SciKit-Learn first


```python
#set up cross validation splits
ts_cv = TimeSeriesSplit(
    n_splits=5,
    gap=48,
    max_train_size=6500,
    test_size=500,
)
```


```python
ols = linear_model.LinearRegression(fit_intercept=False)
cv_results = cross_validate(ols, X, y, cv=ts_cv,
                            scoring=["neg_mean_absolute_error", "neg_root_mean_squared_error", "r2"],
                            return_estimator=True)

mae = -cv_results["test_neg_mean_absolute_error"]
rmse = -cv_results["test_neg_root_mean_squared_error"]
r2 = cv_results["test_r2"]
print(
    f"Mean Absolute Error:     {mae.mean():.3f} +/- {mae.std():.3f}\n"
    f"Root Mean Squared Error: {rmse.mean():.3f} +/- {rmse.std():.3f}\n"
    f"R Squared: {r2.mean():.3f} +/- {rmse.std():.3f}")
```

    Mean Absolute Error:     12.160 +/- 1.099
    Root Mean Squared Error: 15.146 +/- 1.258
    R Squared: 0.363 +/- 1.258


* Look at the coefficients from the cross validation

```python
for reg in cv_results["estimator"]:
    print (reg.coef_)
```

    [[38.8200968   0.36381389 -1.09662218]]
    [[38.79938263  0.33810403 -1.09517704]]
    [[38.59089688  0.34508407 -1.11853572]]
    [[37.06399075  0.40241042 -1.11781534]]
    [[38.80949697  0.35357329 -1.14724296]]


* Now let's use StatsModels


```python
model = sm.OLS(y, X)
results = model.fit()
print(results.summary())
```

                                OLS Regression Results
    ==============================================================================
    Dep. Variable:          SECI_CIMARRON   R-squared:                       0.416
    Model:                            OLS   Adj. R-squared:                  0.416
    Method:                 Least Squares   F-statistic:                     3065.
    Date:                Sat, 26 Oct 2024   Prob (F-statistic):               0.00
    Time:                        15:17:02   Log-Likelihood:                -35562.
    No. Observations:                8592   AIC:                         7.113e+04
    Df Residuals:                    8589   BIC:                         7.115e+04
    Df Model:                           2
    Covariance Type:            nonrobust
    ==============================================================================
                     coef    std err          t      P>|t|      [0.025      0.975]
    ------------------------------------------------------------------------------
    const         40.6287      0.459     88.438      0.000      39.728      41.529
    temp           0.2427      0.015     15.842      0.000       0.213       0.273
    wind          -1.1483      0.015    -76.316      0.000      -1.178      -1.119
    ==============================================================================
    Omnibus:                     3089.212   Durbin-Watson:                   0.209
    Prob(Omnibus):                  0.000   Jarque-Bera (JB):            40420.515
    Skew:                           1.352   Prob(JB):                         0.00
    Kurtosis:                      13.276   Cond. No.                         84.5
    ==============================================================================

    Notes:
    [1] Standard Errors assume that the covariance matrix of the errors is correctly specified.


#### Fitted Values


```python
fig,ax1=plt.subplots()
ax1.plot(y, label='data')
ax1.plot(results.fittedvalues, "r--", label="OLS fit")
ax1.legend(loc='best')
fig.suptitle("OLS Fitted Values")
```

{{< figure src="/Regression_files/Regression_17_1.png"
title="" >}}


```python
compdf = y
compdf.reset_index(drop=True,inplace=True)
compdf['Fitted']=results.fittedvalues.values
sns.scatterplot(data = compdf, x="SECI_CIMARRON", y="Fitted")
plt.gca().axline((0, 0), slope=1, color='red', linestyle='--')
plt.title("Fitted Values Scatter Plot")
plt.show()

```

{{< figure src="/Regression_files/Regression_18_0.png"
title="" >}}


* R2 squared valued is .41, so the model caputres less than half of the variance in the data
* Fitted values show how the model is limited in capturing the high variance priceS
* A linear model may not be the right fit for this data
* Let's dig deeper into evaluating the model and see where we can make improvements

### Evaluating The Regression Model

### Residuals Plots
```python
results.resid.plot()
```


{{< figure src="/Regression_files/Regression_21_1.png"
title="" >}}



```python
#ACF Plot of Residuals
sm.graphics.tsa.plot_acf(results.resid, zero=False, lags=48)
```

{{< figure src="/Regression_files/Regression_22_0.png"
title="" >}}


* Strong autocorrelatoin of the residuals indicatse that our model is inefficient and there is more informatoin to be captured


```python
sns.histplot(results.resid)
```

{{< figure src="/Regression_files/Regression_24_1.png"
title="" >}}

* Residuals are normally distributed which is good

* Combine training data fitted values and residuals into one dataframe
```python
cimarron_res = cimarron_train
cimarron_res["fitted"]=results.fittedvalues.values
cimarron_res["resid"]=results.resid.values
cimarron_res.head()
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
      <th>SECI_CIMARRON</th>
      <th>temp</th>
      <th>wind</th>
      <th>fitted</th>
      <th>resid</th>
    </tr>
    <tr>
      <th>Time</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>2023-01-01 01:00:00-05:00</th>
      <td>17.3592</td>
      <td>4.4</td>
      <td>31.3</td>
      <td>5.755060</td>
      <td>11.604140</td>
    </tr>
    <tr>
      <th>2023-01-01 02:00:00-05:00</th>
      <td>17.1675</td>
      <td>3.5</td>
      <td>28.7</td>
      <td>8.522165</td>
      <td>8.645335</td>
    </tr>
    <tr>
      <th>2023-01-01 03:00:00-05:00</th>
      <td>18.5753</td>
      <td>2.6</td>
      <td>25.7</td>
      <td>11.748589</td>
      <td>6.826711</td>
    </tr>
    <tr>
      <th>2023-01-01 04:00:00-05:00</th>
      <td>18.0899</td>
      <td>3.1</td>
      <td>28.7</td>
      <td>8.425067</td>
      <td>9.664833</td>
    </tr>
    <tr>
      <th>2023-01-01 05:00:00-05:00</th>
      <td>18.7861</td>
      <td>3.1</td>
      <td>28.9</td>
      <td>8.195407</td>
      <td>10.590693</td>
    </tr>
  </tbody>
</table>
</div>




```python
sns.scatterplot(data = cimarron_res, x="temp", y="resid")
plt.show()
```

{{< figure src="/Regression_files/Regression_27_0.png"
title="" >}}


```python
sns.scatterplot(data = cimarron_res, x="wind", y="resid")
plt.show()
```
{{< figure src="/Regression_files/Regression_28_0.png"
title="" >}}


* Residuals plotted against the predictors look non-random
* The plots do show a pttern and the relatoinship maybe nonlinear
* We should try a transformatoin or a nonlinear regression


### Heteroscedasticty
```python
sns.scatterplot(data = cimarron_res, x="fitted", y="resid")
plt.show()
```


{{< figure src="/Regression_files/Regression_30_0.png"
title="" >}}


* the variance clearly increases and shows strong heteroscedasticty

#### Seasonal Dummy Variables: Day of the Week


```python
cimarron_train["DayofWeek"]=cimarron_train.index.dayofweek
cimarron_train.head()
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
      <th>SECI_CIMARRON</th>
      <th>temp</th>
      <th>wind</th>
      <th>DayofWeek</th>
    </tr>
    <tr>
      <th>Time</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>2023-01-01 01:00:00-05:00</th>
      <td>17.3592</td>
      <td>4.4</td>
      <td>31.3</td>
      <td>6</td>
    </tr>
    <tr>
      <th>2023-01-01 02:00:00-05:00</th>
      <td>17.1675</td>
      <td>3.5</td>
      <td>28.7</td>
      <td>6</td>
    </tr>
    <tr>
      <th>2023-01-01 03:00:00-05:00</th>
      <td>18.5753</td>
      <td>2.6</td>
      <td>25.7</td>
      <td>6</td>
    </tr>
    <tr>
      <th>2023-01-01 04:00:00-05:00</th>
      <td>18.0899</td>
      <td>3.1</td>
      <td>28.7</td>
      <td>6</td>
    </tr>
    <tr>
      <th>2023-01-01 05:00:00-05:00</th>
      <td>18.7861</td>
      <td>3.1</td>
      <td>28.9</td>
      <td>6</td>
    </tr>
  </tbody>
</table>
</div>




```python
X=cimarron_train.drop("SECI_CIMARRON", axis="columns")
X = sm.add_constant(X)

y=cimarron_train.drop(["temp", "wind", "DayofWeek"], axis="columns")
y.head()
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
      <th>SECI_CIMARRON</th>
    </tr>
    <tr>
      <th>Time</th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>2023-01-01 01:00:00-05:00</th>
      <td>17.3592</td>
    </tr>
    <tr>
      <th>2023-01-01 02:00:00-05:00</th>
      <td>17.1675</td>
    </tr>
    <tr>
      <th>2023-01-01 03:00:00-05:00</th>
      <td>18.5753</td>
    </tr>
    <tr>
      <th>2023-01-01 04:00:00-05:00</th>
      <td>18.0899</td>
    </tr>
    <tr>
      <th>2023-01-01 05:00:00-05:00</th>
      <td>18.7861</td>
    </tr>
  </tbody>
</table>
</div>




```python
one_hot_encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False, drop= 'first')
ols = linear_model.LinearRegression(fit_intercept=False)

linear_pipeline=make_pipeline(
    ColumnTransformer(
        [('DoW', OneHotEncoder(handle_unknown="ignore", sparse_output=False, drop= 'first'), ['DayofWeek'])],
        remainder='passthrough',
    ),
    ols,
)

cv_results2 = cross_validate(linear_pipeline, X, y, cv=ts_cv, scoring=["neg_mean_absolute_error", "neg_root_mean_squared_error", "r2"], return_estimator=True)
```


```python
mae = -cv_results2["test_neg_mean_absolute_error"]
rmse = -cv_results2["test_neg_root_mean_squared_error"]
r2 = cv_results2["test_r2"]
print(
    f"Mean Absolute Error:     {mae.mean():.3f} +/- {mae.std():.3f}\n"
    f"Root Mean Squared Error: {rmse.mean():.3f} +/- {rmse.std():.3f}\n"
    f"R Squared: {r2.mean():.3f} +/- {rmse.std():.3f}")
```

    Mean Absolute Error:     12.181 +/- 1.099
    Root Mean Squared Error: 15.147 +/- 1.268
    R Squared: 0.363 +/- 1.268



    Original Results without dummy
    Mean Absolute Error:     12.160 +/- 1.099
    Root Mean Squared Error: 15.146 +/- 1.258
    R Squared: 0.363 +/- 1.258



* Let's check the coefficients on one of the estimators


```python
pipe = cv_results2['estimator'][0]
pipe.named_steps['linearregression'].coef_
```




    array([[-1.97916443, -1.04265547,  0.71802774, -0.21776105, -3.30729949,
            -3.32712932, 40.28106137,  0.36309079, -1.10253567]])




    Original Results w/o dummy
    [[38.8200968   0.36381389 -1.09662218]]
    [[38.79938263  0.33810403 -1.09517704]]
    [[38.59089688  0.34508407 -1.11853572]]
    [[37.06399075  0.40241042 -1.11781534]]
    [[38.80949697  0.35357329 -1.14724296]]



* The incercept and the coefficients for wind and temperature are the same for what we got without the dummmy variables for day of week
* Unfortunately this the dummys don't capture anymore infromatoin that without

### Forecasting with Regression


```python
X=cimarron_train.drop(["SECI_CIMARRON", "DayofWeek"], axis="columns")
y=cimarron_train.drop(["temp", "wind", "DayofWeek"], axis="columns")
ols.fit(X, y)
y_pred = ols.predict(cimarron_test.drop(["SECI_CIMARRON"], axis="columns"))
```


```python
plt.figure(figsize=(10,6))
plt.plot(cimarron_test.index, cimarron_test["SECI_CIMARRON"], label = "test", marker = 'o')
plt.plot(cimarron_test.index, y_pred, label = "predict", marker = 'x')
plt.legend()
plt.xticks(rotation=45)
plt.title("OLS Prediction")
plt.show()
```

{{< figure src="/Regression_files/Regression_47_0.png"
title="" >}}

* The model captures the shape of the prices but not the large variance

## Resources
* Hyndman, R.J., & Athanasopoulos, G. (2021) Forecasting: principles and practice, 3rd edition, OTexts: Melbourne, Australia. OTexts.com/fpp3

* Seabold, Skipper, and Josef Perktold. “statsmodels: Econometric and statistical modeling with python.” Proceedings of the 9th Python in Science Conference. 2010.
[Exponential Smoothing Example](https://www.statsmodels.org/stable/examples/notebooks/generated/exponential_smoothing.html#)

* Example adapted from the scikit-learn User Guide: https://scikit-learn.org/stable/user_guide.html.

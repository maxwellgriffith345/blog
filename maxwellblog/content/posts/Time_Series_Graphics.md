+++
title = 'Time Series Graphics'
date = 2024-09-12T17:46:11-05:00
+++

Applying concepts from [Forecasting Principles and Practice](https://otexts.com/fpp3/expsmooth.html) to electricty prices in Python

<!--more-->

## Intro

This is part one in my series to apply the time series forecasting methods presented by Hyndmam and Athanasopoulos in python. I use real world generator location marginal prices that were pulled from South West Power Pool. This section covers the initial scraping of the data and visualizations for time series data

**Concepts**: Seasonal Plots, Scatter Plots, Lag Plots, Autocorrelation.

**Libraries**: Stats Models, Matplotlib, Seaborn, Pandas

## Scrape Southwest Power Pool Day Ahead Location Marginal Prices

* [Grid Status](https://www.gridstatus.io/) provides up to date data on every RTO and they provide a library that can be used to scrape specific data
* awesome site and tool that I recommend playing around with
* [Gridstatus LMP Data Example](https://docs.gridstatus.io/en/latest/Examples/spp/LMP%20Data.html)


```python
import pandas as pd
import gridstatus
```

* Coal Units and Wind Units for a variety
* Wind units should have more negative prices

```python
iso = gridstatus.SPP()

locations = [
"WR.LEC.5",
"WR.JEC.1",
"KCPLIATANUNIAT1",
"KCPLLACYGNEUNLAC1",
"MPS.ROCKCREEK",
"SECI_CIMARRON",
"SECI.KCPS.SPEARVILLE"]
```

* Pull one year of data to start


```python
df_main=iso.get_lmp(date="2023-01-01", end = "2024-01-01", market="DAY_AHEAD_HOURLY")
```


```python
df_main=df_main.loc[df_main['Location'].isin(locations)]
df_main.reset_index(drop=True)
df_main.drop(columns=["Interval Start", "Interval End", "Market", "Location Type", "PNode", "Energy", "Congestion", "Loss"], inplace=True)
df_main.shape
```


```python
df_main.reset_index(drop=True)
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
      <th>Time</th>
      <th>Location</th>
      <th>LMP</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>2023-01-01 00:00:00-06:00</td>
      <td>KCPLIATANUNIAT1</td>
      <td>23.1615</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2023-01-01 00:00:00-06:00</td>
      <td>KCPLIATANUNIAT2</td>
      <td>23.1615</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2023-01-01 00:00:00-06:00</td>
      <td>KCPLLACYGNEUNLAC1</td>
      <td>22.2436</td>
    </tr>
    <tr>
      <th>3</th>
      <td>2023-01-01 00:00:00-06:00</td>
      <td>KCPLLACYGNEUNLAC2</td>
      <td>22.2436</td>
    </tr>
    <tr>
      <th>4</th>
      <td>2023-01-01 00:00:00-06:00</td>
      <td>MPS.ROCKCREEK</td>
      <td>22.8559</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>113875</th>
      <td>2023-12-31 23:00:00-06:00</td>
      <td>WR.JEC.1</td>
      <td>22.7419</td>
    </tr>
    <tr>
      <th>113876</th>
      <td>2023-12-31 23:00:00-06:00</td>
      <td>WR.JEC.2</td>
      <td>22.8056</td>
    </tr>
    <tr>
      <th>113877</th>
      <td>2023-12-31 23:00:00-06:00</td>
      <td>WR.JEC.3</td>
      <td>22.9064</td>
    </tr>
    <tr>
      <th>113878</th>
      <td>2023-12-31 23:00:00-06:00</td>
      <td>WR.LEC.4</td>
      <td>23.1310</td>
    </tr>
    <tr>
      <th>113879</th>
      <td>2023-12-31 23:00:00-06:00</td>
      <td>WR.LEC.5</td>
      <td>23.2291</td>
    </tr>
  </tbody>
</table>
<p>113880 rows × 3 columns</p>
</div>



```python
df_mainPivot = df_main.pivot(index = "Time", columns = "Location", values="LMP")
df_mainPivot.reset_index()
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
      <th>Location</th>
      <th>Time</th>
      <th>KCPLIATANUNIAT1</th>
      <th>KCPLIATANUNIAT2</th>
      <th>KCPLLACYGNEUNLAC1</th>
      <th>KCPLLACYGNEUNLAC2</th>
      <th>MPS.ROCKCREEK</th>
      <th>SECI.KCPS.CIMARRON</th>
      <th>SECI.KCPS.SPEARVILLE</th>
      <th>SECI_CIMARRON</th>
      <th>WR.JEC.1</th>
      <th>WR.JEC.2</th>
      <th>WR.JEC.3</th>
      <th>WR.LEC.4</th>
      <th>WR.LEC.5</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>2023-01-01 00:00:00-06:00</td>
      <td>23.1615</td>
      <td>23.1615</td>
      <td>22.2436</td>
      <td>22.2436</td>
      <td>22.8559</td>
      <td>17.3592</td>
      <td>18.0405</td>
      <td>17.3592</td>
      <td>17.6536</td>
      <td>17.0317</td>
      <td>17.1019</td>
      <td>24.3936</td>
      <td>24.0183</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2023-01-01 01:00:00-06:00</td>
      <td>20.5651</td>
      <td>20.5651</td>
      <td>19.7729</td>
      <td>19.7729</td>
      <td>20.4925</td>
      <td>17.1675</td>
      <td>17.3628</td>
      <td>17.1675</td>
      <td>17.5015</td>
      <td>17.2527</td>
      <td>17.3181</td>
      <td>20.9549</td>
      <td>20.8391</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2023-01-01 02:00:00-06:00</td>
      <td>22.1197</td>
      <td>22.1197</td>
      <td>21.1474</td>
      <td>21.1474</td>
      <td>21.9965</td>
      <td>18.5753</td>
      <td>18.7383</td>
      <td>18.5753</td>
      <td>17.5991</td>
      <td>17.1104</td>
      <td>17.1791</td>
      <td>23.0484</td>
      <td>22.7795</td>
    </tr>
    <tr>
      <th>3</th>
      <td>2023-01-01 03:00:00-06:00</td>
      <td>21.1908</td>
      <td>21.1908</td>
      <td>20.2595</td>
      <td>20.2595</td>
      <td>21.1168</td>
      <td>18.0899</td>
      <td>18.1809</td>
      <td>18.0899</td>
      <td>17.4774</td>
      <td>17.1104</td>
      <td>17.1787</td>
      <td>21.9023</td>
      <td>21.6705</td>
    </tr>
    <tr>
      <th>4</th>
      <td>2023-01-01 04:00:00-06:00</td>
      <td>21.9438</td>
      <td>21.9438</td>
      <td>20.9770</td>
      <td>20.9770</td>
      <td>21.7675</td>
      <td>18.7861</td>
      <td>18.8693</td>
      <td>18.7861</td>
      <td>17.4918</td>
      <td>17.0049</td>
      <td>17.0748</td>
      <td>22.8965</td>
      <td>22.6176</td>
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
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>8755</th>
      <td>2023-12-31 19:00:00-06:00</td>
      <td>25.1442</td>
      <td>25.1442</td>
      <td>24.8819</td>
      <td>24.8819</td>
      <td>16.9229</td>
      <td>22.7914</td>
      <td>23.1359</td>
      <td>22.7914</td>
      <td>23.9770</td>
      <td>24.0462</td>
      <td>24.1494</td>
      <td>24.7306</td>
      <td>24.8162</td>
    </tr>
    <tr>
      <th>8756</th>
      <td>2023-12-31 20:00:00-06:00</td>
      <td>25.0533</td>
      <td>25.0533</td>
      <td>24.5311</td>
      <td>24.5311</td>
      <td>24.7022</td>
      <td>23.3450</td>
      <td>23.6479</td>
      <td>23.3450</td>
      <td>23.8784</td>
      <td>23.9503</td>
      <td>24.0563</td>
      <td>24.4739</td>
      <td>24.5689</td>
    </tr>
    <tr>
      <th>8757</th>
      <td>2023-12-31 21:00:00-06:00</td>
      <td>24.9286</td>
      <td>24.9286</td>
      <td>23.9259</td>
      <td>23.9259</td>
      <td>24.8643</td>
      <td>23.3814</td>
      <td>23.6222</td>
      <td>23.3814</td>
      <td>23.6232</td>
      <td>23.6966</td>
      <td>23.8023</td>
      <td>24.2121</td>
      <td>24.2862</td>
    </tr>
    <tr>
      <th>8758</th>
      <td>2023-12-31 22:00:00-06:00</td>
      <td>26.5195</td>
      <td>26.5195</td>
      <td>25.5493</td>
      <td>25.5493</td>
      <td>26.4230</td>
      <td>25.4710</td>
      <td>25.6972</td>
      <td>25.4710</td>
      <td>25.2807</td>
      <td>25.3562</td>
      <td>25.4686</td>
      <td>25.8145</td>
      <td>25.9067</td>
    </tr>
    <tr>
      <th>8759</th>
      <td>2023-12-31 23:00:00-06:00</td>
      <td>23.6572</td>
      <td>23.6572</td>
      <td>23.1228</td>
      <td>23.1228</td>
      <td>23.4068</td>
      <td>23.2068</td>
      <td>23.4067</td>
      <td>23.2068</td>
      <td>22.7419</td>
      <td>22.8056</td>
      <td>22.9064</td>
      <td>23.1310</td>
      <td>23.2291</td>
    </tr>
  </tbody>
</table>
<p>8760 rows × 14 columns</p>
</div>




```python
df_mainPivot.columns.name=''
```


```python
df_mainPivot.index
```




    DatetimeIndex(['2023-01-01 00:00:00-06:00', '2023-01-01 01:00:00-06:00',
                   '2023-01-01 02:00:00-06:00', '2023-01-01 03:00:00-06:00',
                   '2023-01-01 04:00:00-06:00', '2023-01-01 05:00:00-06:00',
                   '2023-01-01 06:00:00-06:00', '2023-01-01 07:00:00-06:00',
                   '2023-01-01 08:00:00-06:00', '2023-01-01 09:00:00-06:00',
                   ...
                   '2023-12-31 14:00:00-06:00', '2023-12-31 15:00:00-06:00',
                   '2023-12-31 16:00:00-06:00', '2023-12-31 17:00:00-06:00',
                   '2023-12-31 18:00:00-06:00', '2023-12-31 19:00:00-06:00',
                   '2023-12-31 20:00:00-06:00', '2023-12-31 21:00:00-06:00',
                   '2023-12-31 22:00:00-06:00', '2023-12-31 23:00:00-06:00'],
                  dtype='datetime64[ns, US/Central]', name='Time', length=8760, freq=None)




```python
df_mainPivot.columns
```




    Index(['KCPLIATANUNIAT1', 'KCPLIATANUNIAT2', 'KCPLLACYGNEUNLAC1',
           'KCPLLACYGNEUNLAC2', 'MPS.ROCKCREEK', 'SECI.KCPS.CIMARRON',
           'SECI.KCPS.SPEARVILLE', 'SECI_CIMARRON', 'WR.JEC.1', 'WR.JEC.2',
           'WR.JEC.3', 'WR.LEC.4', 'WR.LEC.5'],
          dtype='object', name='')




```python
df_mainPivot.to_csv("LMPSWide3years.csv")
```

## Time Series Graphics


```python
import pandas as pd
import numpy as np
import seaborn as sns
%matplotlib inline
import matplotlib.pyplot as plt
```


```python
df=pd.read_csv("LMPSWide2023.csv")
df["Time"]=df["Time"].str[:16]
df['Time'] = pd.to_datetime(df['Time'], format='%Y-%m-%d %H:%M')
df.set_index('Time', inplace=True)
```


```python
"""
"WR.LEC.5",
"WR.JEC.1",
"KCPLIATANUNIAT1",
"KCPLLACYGNEUNLAC1",
"MPS.ROCKCREEK",
"SECI_CIMARRON",
"SECI.KCPS.SPEARVILLE"
"""
df.drop(columns=["KCPLIATANUNIAT2","KCPLLACYGNEUNLAC2","SECI.KCPS.CIMARRON", "WR.LEC.4", "WR.JEC.2", "WR.JEC.3", "SECI.KCPS.CIMARRON"], inplace=True)
```


```python
fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12,10))
df.plot(ax=ax1)
df.loc["2023-07-01":"2023-08-31"].plot(ax=ax2, legend=False)
df.loc["2023-03-01":"2023-03-31"].plot(ax=ax3, legend=False)
df.loc["2023-03-01":"2023-03-08"].plot(ax=ax4, legend=False)
ax1.set_title("2023 Coal and Wind LMPS")
ax2.set_title("July through August LMPS")
ax3.set_title("March LMPS")
ax4.set_title("One Week LMPS")

for ax in fig.get_axes():
    ax.set(xlabel='', ylabel='LMP')

ax1.legend(fancybox=True, framealpha=0.5)
fig.tight_layout()
plt.show()
```


{{< figure src="/Time_Series_Graphics_files/Time_Series_Graphics_20_0.png"
title="Year, Season, Month, Week LMPS" >}}


* What stands out immediedtly are the big price speaks in the summer, and the increase in negative price volatility in the fall
Additionally, there seems to be a general lift in prices from May through August.
* With one year it is hard to tell if there is an overall trend in the prices
Seasonal-both the price level and the volatility in prices show seasonal patters

### Seasonal Plots
Stacking the weeks


```python
coalgens = ["KCPLIATANUNIAT1", "KCPLLACYGNEUNLAC1", "WR.JEC.1", "WR.LEC.5"]
windgens = ["MPS.ROCKCREEK", "SECI.KCPS.SPEARVILLE", "SECI_CIMARRON"]

df['DayOfWeek']=df.index.day_name()
df['TimeOfDay']=df.index.time
df['WekOfYear']= df.index.isocalendar().week
df['DayTime']= df.index.strftime('%A %H:%M')

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
      <th>KCPLIATANUNIAT1</th>
      <th>KCPLLACYGNEUNLAC1</th>
      <th>MPS.ROCKCREEK</th>
      <th>SECI.KCPS.SPEARVILLE</th>
      <th>SECI_CIMARRON</th>
      <th>WR.JEC.1</th>
      <th>WR.LEC.5</th>
      <th>DayOfWeek</th>
      <th>TimeOfDay</th>
      <th>WekOfYear</th>
      <th>DayTime</th>
    </tr>
    <tr>
      <th>Time</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
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
      <th>2023-01-01 00:00:00</th>
      <td>23.1615</td>
      <td>22.2436</td>
      <td>22.8559</td>
      <td>18.0405</td>
      <td>17.3592</td>
      <td>17.6536</td>
      <td>24.0183</td>
      <td>Sunday</td>
      <td>00:00:00</td>
      <td>52</td>
      <td>Sunday 00:00</td>
    </tr>
    <tr>
      <th>2023-01-01 01:00:00</th>
      <td>20.5651</td>
      <td>19.7729</td>
      <td>20.4925</td>
      <td>17.3628</td>
      <td>17.1675</td>
      <td>17.5015</td>
      <td>20.8391</td>
      <td>Sunday</td>
      <td>01:00:00</td>
      <td>52</td>
      <td>Sunday 01:00</td>
    </tr>
    <tr>
      <th>2023-01-01 02:00:00</th>
      <td>22.1197</td>
      <td>21.1474</td>
      <td>21.9965</td>
      <td>18.7383</td>
      <td>18.5753</td>
      <td>17.5991</td>
      <td>22.7795</td>
      <td>Sunday</td>
      <td>02:00:00</td>
      <td>52</td>
      <td>Sunday 02:00</td>
    </tr>
    <tr>
      <th>2023-01-01 03:00:00</th>
      <td>21.1908</td>
      <td>20.2595</td>
      <td>21.1168</td>
      <td>18.1809</td>
      <td>18.0899</td>
      <td>17.4774</td>
      <td>21.6705</td>
      <td>Sunday</td>
      <td>03:00:00</td>
      <td>52</td>
      <td>Sunday 03:00</td>
    </tr>
    <tr>
      <th>2023-01-01 04:00:00</th>
      <td>21.9438</td>
      <td>20.9770</td>
      <td>21.7675</td>
      <td>18.8693</td>
      <td>18.7861</td>
      <td>17.4918</td>
      <td>22.6176</td>
      <td>Sunday</td>
      <td>04:00:00</td>
      <td>52</td>
      <td>Sunday 04:00</td>
    </tr>
  </tbody>
</table>
</div>




```python
ticksdays =[0]+[((24*i)-1) for i in range (1,7)]
```


```python
#Coal Gens
coalgens = ["KCPLIATANUNIAT1", "KCPLLACYGNEUNLAC1", "WR.JEC.1", "WR.LEC.5"]
unique_days=df['DayOfWeek'].unique()
fig, axs = plt.subplots(2, 2, sharex=True, sharey=True, figsize=(10,10))
fig.suptitle('LMPS by Week for Coal Generators')
for gen,axs in zip(coalgens, fig.get_axes()):
    sns.lineplot(data=df, x='DayTime', y=gen, hue='WekOfYear', errorbar=None, legend=False, ax = axs)
    axs.set_xticks(ticks=ticksdays, labels=unique_days, rotation=45)
    axs.set_title(gen)

for ax in fig.get_axes():
    ax.set(xlabel='', ylabel='LMP')

# Hide x labels and tick labels for top plots and y ticks for right plots.
for ax in fig.get_axes():
    ax.label_outer()
fig.tight_layout()
```

{{< figure src="/Time_Series_Graphics_files/Time_Series_Graphics_27_0.png"
title="Weekly LMP Shapes for Coal" >}}

* Prices drop off in the overnight hours, and spike in the morning and afternoon.
* There are negative prices almost exclusivly at night. Large spikes on Monday, Tuesday, and Friday.
* There is some varince in prices. spread of the prices over the weeks but they follow a closesly grouped pattern of movement.


```python
#Wind Gens
windgens = ["MPS.ROCKCREEK", "SECI.KCPS.SPEARVILLE", "SECI_CIMARRON"]
unique_days=df['DayOfWeek'].unique()
fig, axes = plt.subplots(2, 2, sharex=True, sharey= True, figsize=(10,10))
fig.suptitle('LMPS by Week for Wind Generators')
for gen,axs in zip(windgens, fig.get_axes()):
    sns.lineplot(data=df, x='DayTime', y=gen, hue='WekOfYear', errorbar=None, legend=False, ax = axs, palette= "mako" )
    axs.set_xticks(ticks=ticksdays, labels=unique_days, rotation=45)
    axs.set_title(gen)

axes[1,1].axis("off")

for ax in fig.get_axes():
    ax.set(xlabel='', ylabel='LMP')

fig.tight_layout()
# Hide x labels and tick labels for top plots and y ticks for right plots.
#for ax in fig.get_axes():
   # ax.label_outer()
```


{{< figure src="/Time_Series_Graphics_files/Time_Series_Graphics_29_0.png"
title="Weekly LMP Shapes for Wind" >}}

* Larger swings in day night pricing when compared to the coal geneartors.
* Across the weeks there is more variance, the prices are less tightly grouped.
* This could indicate large pricing trends depending on the season. Which would align the assumption that fall and spring have the most hours of wind generation.
* More frequent and deeper negative price trends.
* A similar pattern of large spikes on Monday Tuesday, Wednesday, and Friday.

### Scatter Plots


```python
#add a season column to label the dots by season
seasons = {12: "Winter", 1: "Winter", 2: "Winter",
           3: "Spring", 4:"Spring", 5: "Spring",
          6: "Summer", 7: "Summer", 8: "Summer",
          9: "Fall", 10: "Fall", 11: "Fall"}
df_season=df.assign(Season = df.index.month.map(seasons) )

sns.pairplot(df_season, corner = True, hue= "Season")
```


{{< figure src="/Time_Series_Graphics_files/Time_Series_Graphics_33_1.png"
title="Pair Plots" >}}


* Coal units are highly linearly related, the wind units are less so. The summer prices show the most variation.

### Lag Plots


```python
# Cimarron Wind
lags = [i for i in range (1,25)]
fig, axs = plt.subplots(6, 4, sharex=True, sharey=True, figsize=(12,10))
for lagstep, axs in zip(lags,fig.get_axes()):
    pd.plotting.lag_plot(df["SECI_CIMARRON"], lag=lagstep, ax=axs, marker = '+')
    if lagstep <21:
        axs.set_xlabel('')
fig.suptitle("Cimarron Wind LMP Lag Plots")
```

{{< figure src="/Time_Series_Graphics_files/Time_Series_Graphics_36_1.png"
title="Lag Plots Cimarron Wind" >}}



```python
# Iatan1 Coal
fig, axs = plt.subplots(6, 4, sharex=True, sharey=True, figsize=(12,10))
for lagstep, axs in zip(lags,fig.get_axes()):
    pd.plotting.lag_plot(df["KCPLIATANUNIAT1"], lag=lagstep, ax=axs, marker = '+')
    if lagstep <21:
        axs.set_xlabel('')
fig.suptitle("Iatan1 Coal LMP Lag Plots")
```

{{< figure src="/Time_Series_Graphics_files/Time_Series_Graphics_37_1.png"
title="Lag Plots Iatan1 Coal" >}}

* For both wind and coal generators there is a week positive relation for one and two hour lags.
* Three hour lag shows no relationship.
* Then a negative relationship begins to form until lags nine through 16 where the relationship is inverted.
* This would make sense as the over night hours vs daytime hours would be opposite peaks for low and high prices.
* From sevent to twenty one there is a weak negative relationsip and lags twenty three and twenty four show no relationship.

### Autocorrelation
Measures the linear relationship between lagged values of a time series


```python
import statsmodels.api as sm

fig, axs = plt.subplots(4, 2, sharex=True, sharey=True, figsize=(10,10))
fig.suptitle('Autocorrelation for LMP over 2 days')
for col, axs in zip(list(df.columns),fig.get_axes()):
    sm.graphics.tsa.plot_acf(df[col], lags=48, zero=False, ax=axs, title = col+' Autocorrelation')
```

{{< figure src="/Time_Series_Graphics_files/Time_Series_Graphics_42_0.png"
title="AutoCorrelation" >}}


* Positive Autocorrelatoin: high values of Yt predicut future high values for Yt+h, low values of Yt predict future low values for Yt+h
* Negative autocorrelation: Yt has immmediate reversals in adjacent periods
* Bartlett Condifence Bands: if autocorrelation falls outside the shaded reiong, it is statistically different than zero.

* For the above auto correlation of Cimarron, we see that there is a periodic trend for prices from the 24 hours before and 48 hours before.
*   What the price was at that time yesterday day and 2 days ago is a good indication of what it will be now.
*   We see a insignificant autocorrecations at 12 hours and 35 hours suggesting those lags have little to tell us about what the price is at the current time


## Resources
* Hyndman, R.J., & Athanasopoulos, G. (2021) Forecasting: principles and practice, 3rd edition, OTexts: Melbourne, Australia. OTexts.com/fpp3

* Seabold, Skipper, and Josef Perktold. “statsmodels: Econometric and statistical modeling with python.” Proceedings of the 9th Python in Science Conference. 2010.
[Exponential Smoothing Example](https://www.statsmodels.org/stable/examples/notebooks/generated/exponential_smoothing.html#)

* Hansen, Bruce (2018) [Economic Forecasting University of Wisconsin Madison](https://users.ssc.wisc.edu/~bhansen/460/) Notes on interpreting autocorrelation


## Comparision of two data-driven approaches for long-term time series flood forecasting with a limited dataset - LSTM vs ARIMA  

>This repository documents the development of an LSTM and ARIMA model in MATLAB.  

### ___Data acquisition:___

Time series of daily flow data at station 28074 Kegworth, River Soar.  
Link to source/download: [NRFA, 2022](https://nrfa.ceh.ac.uk/data/station/meanflow/28082)

- [x] There are 84 observations missing between 07/02/1984 to 01/03/1991. 
- [x] Since this period is large (> 3 years) classic data implementation methods such as linear interpolation was not employed as it assumes explicit linearity between observations. 

The time series used therefore began from 01/03/1991 to 30/09/2020, the latest measurement at the time.  

---

### ___Procedure:___
The steps to develop the models can essentially be broken down into **7** key stages outlined as follows: 

*#1* :mag: Selection of relevant input variables via exploratory time series analysis (Ljung-Box test for autocorrelation, modified Mann-Kendall trend test...)

*#2* :wrench: Time series reconstruction 

![image](https://user-images.githubusercontent.com/86715613/174202814-9b19394b-5f36-401f-ba6c-1ddffd89f5cd.png)

Step-by-step of algorithm developed for time series reconstruction for month, _i_.  
Where _Q<sub>i, j</sub>_ is the observation on day _j_ of the _i<sup>th</sup>_ month. 
1. First, the rolling 7-day sums of river flow volume, was calculated within month, _i_. The largest 7-day river flow volume within that month, _Q<sub>i, max</sub>_, is then the _i<sup>th</sup>_ observation of the new time series. 
2. This process was repeated for all 355 months i.e., until _i=355_. 

Note: The number of days in a given month ranges from 28 to 31, the example above, demonstrates the algorithm on a month with 30 days, but it is applicable to all month sizes.

*#3* :chart_with_downwards_trend: Stationary characteristic analysis (SSA time series decomposition, modified Leybourne-McCabe stationary test...)

![image](https://user-images.githubusercontent.com/86715613/174203743-eb4c6856-4b04-4eb1-999b-6f75f2715ee3.png)

The SSA decomposition of monthly maximum 7-day river flow volume time series into long-term trend, seasonal and irregular components. Components with nonnegative values (a) and (c) show the same data range to aid comparisons.

*#4* :computer: Pre-processing (differencing, partioning, normalisation)  

![image](https://user-images.githubusercontent.com/86715613/174201567-ed8cf551-27ce-4000-a7b3-0dcac0afa307.png)

The reconstructed time series above shows the pre-processing steps at each stage. 
The calibration period the first 90% of the time series was used to calibrate the ARIMA and LSTM models and the remaining 10%, highlighted in red from month 319 to 355 inclusive, was the forecasting period.

*#5* :brain: LSTM calibration 

Hyperparameters of the LSTM were adjusted via Bayesian Optimisation to avoid over-fitting. After, 50 trials the combination of hyperparameters that produced the lowest RMSE (see Section 4.3.2) was used for the final LSTM neural network configuration.  
The 'Adam' optimisation algorithm was also implemented to improve LSTM performance.

*#6* :chart_with_upwards_trend: ARIMA calibration (BIC, ACF and PACF)

![image](https://user-images.githubusercontent.com/86715613/174204833-8af50cce-7c1f-4969-8a8e-0e4e9f9cdb71.png)

A total of 16 autoregressive model combinations models each of which were fitted to the time series, where 1≤p≤4 and 1≤q≤4. 
Once fitted, the BIC was calculated for each mode.  
The model which minimised the BIC was deemed to be of the best fit.
The BIC values were used in conjunction with the ACF and PACF correlograms shown above to verify the BIC results. (a) ACF correlogram (b) PACF correlogram. The significance bounds of both correlograms are at ±5% of the zero line 

*#7* :bar_chart: Model performance evaluation (R<sup>2</sup>, modified Nash-Sutcliffe index...)

![image](https://user-images.githubusercontent.com/86715613/174205478-a5113b18-cb2a-4e00-8ca4-8e0618506e76.png)

(a) R<sup>2</sup>, (b) MNS and (c) RMSE from forecast periods 1 to 6 for LSTM and ARIMA(4, 1, 4) models. The results of the ARIMA(4, 1, 4) are orange and LSTM are in purple.

>All of the above steps steps are condensed in the flow chart below: 

![Flowchart of procedure](https://user-images.githubusercontent.com/86715613/174206891-002712c0-1429-4b67-9025-0abbf9a726a4.png)

Further assessment of model bias and variance were also conducted formally via the Shapiro-Wilk test and graphically via Q-Q plots and histograms.

---
### ___Helpful reading:___
>Havard references to some of key research papers that aided this project in no particular order :open_book::

| No. |  Paper/Book | 
| --- |:----------:| 
|1    |Kratzert, F. et al. (2019) ‘NeuralHydrology – Interpreting LSTMs in Hydrology’, Lecture Notes in Computer Science (including subseries Lecture Notes in Artificial Intelligence and Lecture Notes in Bioinformatics), 11700 LNCS, pp. 347–362. doi:10.1007/978-3-030-28954-6_19.|
|2    |Le, X.H. et al. (2019) ‘Application of Long Short-Term Memory (LSTM) Neural Network for Flood Forecasting’, Water 2019, Vol. 11, Page 1387, 11(7), p. 1387. doi:10.3390/W11071387|
|3    |Sahoo, B.B., Jha, R., Singh, A. and Kumar, D. (2019) ‘Long short-term memory (LSTM) recurrent neural network for low-flow hydrological time series forecasting’, Acta Geophysica, 67(5), pp. 1471–1481. doi:10.1007/S11600-019-00330-1/FIGURES/9.|
|4  |Box, G.E.P., Jenkins, G.M. and Reinsel, G.C. (1994) ‘Analysis of seasonal time series’, in Time series analysis : forecasting and control. 3rd ed. Englewood Cliffs  N.J.: Prentice Hall, p. 306.|
|5  |Yu, Z. et al. (2017) ‘ARIMA modelling and forecasting of water level in the middle reach of the Yangtze River’, 2017 4th International Conference on Transportation Information and Safety, ICTIS 2017 - Proceedings, pp. 172–177. doi:10.1109/ICTIS.2017.8047762.|
|6  |Srivastava, N. et al. (2014) ‘Dropout: A Simple Way to Prevent Neural Networks from Overfitting’, Journal of Machine Learning Research, 15, pp. 1929–1958|
|7  |Golyandina, N. and Zhigljavsky, A. (2013) Singular Spectrum Analysis for Time Series. Berlin, Heidelberg: Springer Berlin Heidelberg (SpringerBriefs in Statistics). doi:10.1007/978-3-642-34913-3|
|8  |Berardi, V.L. and Zhang, G.P. (2003) ‘An empirical investigation of bias and variance in time series forecasting: Modeling considerations and error evaluation’, IEEE Transactions on Neural Networks, 14(3), pp. 668–679. doi:10.1109/TNN.2003.810601|
|9  |Hamed, K.H. and Ramachandra Rao, A. (1998) ‘A modified Mann-Kendall trend test for autocorrelated data’, Journal of Hydrology, 204(1–4), pp. 182–196. doi:10.1016/S0022-1694(97)00125-X.|
|10 |Legates, D.R. and McCabe, G.J. (1999) ‘Evaluating the use of “goodness-of-fit” measures in hydrologic and hydroclimatic model validation’, Water Resources Research, 35(1), pp. 233–241. doi:10.1029/1998WR900018/FORMAT/PDF|
|11 |Leybourne, S.J. and HcCabe, B.P.M. (1999) ‘Modified stationarity tests with data-dependent model-selection rules’, Journal of Business and Economic Statistics, 17(2), pp. 264–270. doi:10.1080/07350015.1999.10524816.|


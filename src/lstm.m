%% INSTALL to work on MATLAB version 2021a and 2021b: 
% Statistics and Machine Learning Toolbox
% Econometrics Toolbox

%% Stage 0: Time series characteristics of daily precipitation & flow 
% Plot of precipitation and flow on the same graph.
table_kegworth_flow = array2table(kegworth_flow); 

time = datetime(table_kegworth_flow.kegworth_flow3,...
    table_kegworth_flow.kegworth_flow2,...
    table_kegworth_flow.kegworth_flow1);

figure
plot(time, table_kegworth_flow.kegworth_flow4);
xlabel('$Time$ $(Days)$', 'Interpreter','latex');
ylabel('$Flow$ $(m^{3}/s$)', 'Interpreter','latex'); 
set(gca, 'FontName', 'Cambria Math');
hold on 
yyaxis right
plot(time, kegworth_precip.Precipitation);
set(gca, 'YDir','reverse');
ylabel('$Precipitation$ $(mm)$', 'Interpreter','latex'); 
ylim([0 200]); 
legend('$Flow$', '$Precipitation$', 'Interpreter','latex');


% Conduct Ljung-Box test to check for autocorrelation for flow...
% @ alpha = 0.05.
res_flow = table_kegworth_flow.kegworth_flow4 -...
    mean(table_kegworth_flow.kegworth_flow1);
%[h1_flow, pValue_flow, stat_flow, cValue_flow] =...
%    lbqtest(res_flow, 'Lags', 20); 
%disp('Daily flow series test for autocorrelation:')
%disp(h1_flow); 
%disp(pValue_flow); 
%disp(stat_flow); 
%disp(cValue_flow); 

% Conduct Ljung-Box test to check for autocorrelation for...
% precipitation @ alpha = 0.05.

res_precip = kegworth_precip.Precipitation -...
    mean(kegworth_precip.Precipitation);
%[h1_precip, pValue_precip, stat_precip, cValue_precip] =...
%    lbqtest(res_precip, 'Lags', 20);
%disp('Daily precipitation series test for autocorrelation:')
%disp(h1_precip); 
%disp(pValue_precip); 
%disp(stat_precip); 
%disp(cValue_precip); 

% Call the modified Mann-Kendall test for flow (mmtest).
%[H, pp_value, test_stat, crit] = mmtest(res_flow, 0.05); 
%disp('Modified Mann-Kendall test results for flow:')
%disp(H);
%disp(pp_value);
%disp(abs(test_stat)); 
%disp(crit); 

% Call the modified Mann-Kendall test for precip (mmtest).
%[H_2, pp_value_2, test_stat_2, crit_2] = mmtest(res_precip, 0.05); 
%disp('Modified Mann-Kendall test results for precipitation:')
%disp(H_2);
%disp(pp_value_2);
%disp(abs(test_stat_2)); 
%disp(crit_2); 

% Sen slope
%disp(Sen_Slope(res_precip));
%disp(Sen_Slope(res_flow));

% Plot precip against flow 
figure 
scatter(res_precip, res_flow); 
xlabel('$Precipitation$ $(mm)$', 'Interpreter','latex');
ylabel('$Flow$ $(m^{3}/s$)', 'Interpreter','latex'); 
set(gca, 'FontName', 'Cambria Math');
title('$R = 0.3315$ and $\tau = 0.2632$', 'Interpreter','latex'); 
xlim([0 50]); 
ylim([0 700]);
hold on 
plot(xlim, ylim, 'LineStyle', '--'); 
legend('', '$y = x$', 'Interpreter','latex');

figure
plotResiduals(fitlm(res_precip, res_flow), 'fitted');
 
% Correlation coefficient
[rho, pvalue] = corr(res_precip, res_flow, 'Type', 'Kendall');  
disp(rho); 
disp(pvalue); 

% Compute Pearson's correlation coefficients between flow and...
% precipitation
%rho = corrcoef(table_kegworth_flow.kegworth_flow4,...
%    kegworth_precip.Precipitation);
%disp('PMCC:')
%disp(rho); % Low value 0.3315 indicates weak correlation, therefore...
% precipitation was not considered at a unique input variable +...
% the results of the MK test show neither exhibit a trend over time +...
% precipitation more strongly than flow. 

% Kendall's rank correlation between flow and precipitation.
%tau = corr(res_flow, res_precip,'type', 'Kendall'); 
%disp('Kendalls rank correlation coefficient:'); 
%disp(tau);

%% Stage 1a: Re-format original Kegworth flow series.
% Start time series after 7 year gap.
completed_keg = kegworth(4474:15280, :); 

% Convert table to array.
a = table2array(completed_keg); 

% Calculate rolling total of every value.
window = 6; % window in movmean function is window + 1
total = [a(:, 1), a(:,2), a(:, 3), movmean(a(:,4), [0 window])*7];  

% Convert array to table and give column headings.
final_keg = array2table(total); 
final_keg.Properties.VariableNames{1} = 'Days';
final_keg.Properties.VariableNames{2} = 'Months';
final_keg.Properties.VariableNames{3} = 'Years';
final_keg.Properties.VariableNames{4} = 'Flows';

% Find maximum discharge of every month.
tmaxVals=rowfun(@max,final_keg,'InputVariables','Flows', ...
                          'GroupingVariables',{'Years','Months'}, ...
                          'OutputVariableNames',{'GroupMax'});

% Initialise the number of months 
months = 1:length(tmaxVals.GroupMax); 

% Plot maximum 7 day total flow against months forming new time series...
% and mean.
figure
plot(months, tmaxVals.GroupMax, 'LineWidth', 0.6); 
xlabel('$Month$', 'Interpreter','latex');
ylabel('$Flow$ $(m^{3}/s)$', 'Interpreter','latex');
set(gca, 'FontName', 'Cambria Math');
legend('$Monthly$ $maximum$ $7-day$ $cumulative$ $flow$', 'Interpreter','latex');
xlim([0 355]);



%% Stage 2a: Determine optimal lag number via BIC methodology.
% Run only on raw time series: 'minP' = 11

% (1) Create zeros matrix to store all 16 combinations of ARIMA models...
%LogL = zeros(4,4); 
%PQ = zeros(4,4);
%for p = 1:4
%    for q = 1:4
%        Mdl = arima(p, 0, q); 
%        [EstMdl,~,LogL(p, q)] = estimate(Mdl, tmaxVals.GroupMax,...
%            'Display', 'off');
%        PQ(p, q) = p + q;
%     end
%end

% (2) Calculate BIC per combination: 
%logL = LogL(:);
%pq = PQ(:);
%[~,bic] = aicbic(logL, pq + 1, 355);
%BIC = reshape(bic, 4, 4); 

% (3) Obtain the optimal number of lags by finding minimum BIC value:
%minBIC = min(BIC,[],'all');
%minP = find(minBIC == BIC); 

%% Stage 2b: ITERATIVE : Conduct Leybourne-McCabe test for stationarity.

% (4) Input 'minP' into modified LM test to check for stationarity
% If h = 0 then accept the null hypothesis. If h = 1 reject null.
%[h, p_value, ~, ~] = lmctest(tmaxVals.GroupMax,...
%   "trend", false, "Lags", 12, "Test", "var2", "alpha", 0.05);
%disp(h);
%disp(p_value);
% If time series is stationary continue. 
% Repeat this stage at Lag 12 until null is not rejected.
% RESULTS: Differencing order 1 required. Therefore ARIMA(4, 1, 4).

%% Stage 3a: Decompose time series using SSA.

[trend, seasonal, irregular] = trenddecomp(tmaxVals.GroupMax, ...
    "ssa", (355-1)/2, NumSeasonal=1);
% The lag value must be a positive real scalar between 3 and N/2...
% where N is the number of elements in the first input.
% Therefore, 355 - 1/2 =  N 

% Long-term trend component: 
figure 
subplot(3,1,1);
plot(months, trend,'black'); 
xlabel('$Month$','Interpreter','latex');
ylabel('$T_t$','Interpreter','latex');
set(gca, 'FontName', 'Cambria Math');
xlim([0 355]);
ylim([0 400]); 

% Seasonal component (used to support ACF plot results and model):
subplot(3, 1, 2);
plot(1:355, seasonal,'black'); 
xlabel('$Month$','Interpreter','latex');
ylabel('$S_t$','Interpreter','latex');
set(gca, 'FontName', 'Cambria Math');
xlim([0 355]);
ylim([-100 100])

% Irregular component 
subplot(3, 1, 3);
plot(1:355, irregular,'black'); 
xlabel('$Month$','Interpreter','latex');
ylabel('$I_t$','Interpreter','latex');
set(gca, 'FontName', 'Cambria Math');
xlim([0 355]);


%% LZ complexity
quant = quantizer('double');
binseq = num2bin(quant, transpose(irregular)); 
binseq_vector = str2num(binseq); 

[C, ~, ~] = lz(binseq_vector, 'primitive', 1); 

%% Stage 4: Check estimated optimal model with ACF and PACF + seasonality.
% Calculate ACF and PACF values to 50 lags.
lags = (0:1:50); 
auto = autocorr(tmaxVals.GroupMax, 'NumLags', length(lags)-1); 
parauto = parcorr(tmaxVals.GroupMax, 'NumLags', length(lags)-1);

% Set bounds at 95% confidence interval.
upper_bound = 0.05 + zeros(1, length(lags)); 
lower_bound = -0.05 + zeros(1, length(lags)); 

% 1st plot: ACF w/ bounds.
figure
subplot(2, 1, 1) 
stem(lags, auto, 'black'); 
xlabel('Lag', 'Interpreter','latex'); 
ylabel('ACF', 'Interpreter','latex');
set(gca, 'FontName', 'Cambria Math');
hold on 
plot(lags, upper_bound, '--', 'Color', 'r'); % Plot upper bound
hold on
plot(lags, lower_bound, '--', 'Color', 'r'); % Plot lower bound 
xlim([0 50]);

% 2nd plot: PACF w/ bounds. 
subplot(2, 1, 2)
stem(lags, parauto, 'black'); 
xlabel('Lag', 'Interpreter','latex'); 
ylabel('PACF', 'Interpreter','latex');
hold on 
plot(lags, upper_bound,  '--', 'Color', 'r'); % Plot upper bound
hold on
plot(lags, lower_bound,  '--', 'Color', 'r'); % Plot lower bound 
xlim([0 50]);


%%
% Remove the noise component of the time series.
% Seasonaility is of interest to describe pattern of flood volumes.
time_series_RAW = tmaxVals.GroupMax; 
time_series = time_series_RAW - irregular - trend; 

% Split training and forecasting data.
TRAIN_RAW = tmaxVals.GroupMax(1:319);
TRAIN = TRAIN_RAW - irregular(1:319) - trend(1:319); 

% Do the same to FORECAST.
FORECAST_RAW = tmaxVals.GroupMax(320:355);
FORECAST = FORECAST_RAW - irregular(320:355) - trend(320:355); 



%% Stage 3b: Difference the time series. 
diff_TRAIN = diff(TRAIN, 1);

%% Stage 3b: Normalise TRAIN and FORECAST using diff_TRAIN min and max values.
N_TRAIN = min_max(diff_TRAIN, diff_TRAIN);
N_FORECAST = min_max(FORECAST, diff_TRAIN);

%% Stage 3d: Use p and q values to estimate an ARIMA model. 
arima_model = arima(4,1,4); 
arima_model.Distribution = 'Gaussian'; 
est_arima_model = estimate(arima_model, N_TRAIN, 'Display', 'off');

residuals_TRAIN = infer(est_arima_model, N_TRAIN);
A_TRAIN = N_TRAIN - residuals_TRAIN; % Results of ARIMA estimated model

residuals_FORECAST = infer(est_arima_model, N_FORECAST);
A_FORECAST = N_FORECAST - residuals_FORECAST;

%% Stage 3e: Denormalise A_TRAIN time series. 
diff_TRAIN = inv_min_max(N_TRAIN, diff_TRAIN); 
A_TRAIN = inv_min_max(A_TRAIN, diff_TRAIN); 

diff_FORECAST = inv_min_max(N_FORECAST, diff_TRAIN); 
A_FORECAST = inv_min_max(A_FORECAST, diff_TRAIN);

%% Stage 3f: Reverse first order differencing by just using TRAIN.
% TRAIN is before differencing and normalisation. 
A_TRAIN(end+1) = TRAIN(end);
A_TRAIN = conv(A_TRAIN, [1 1], 'valid');

A_FORECAST(end+1) = FORECAST(end);
A_FORECAST = conv(A_FORECAST, [1 1], 'valid');


%% Stage 3f: Add the trend and seasonal components of TRAIN and A_TRAIN.
TRAIN = TRAIN + irregular(1:319) + trend(1:319); 
A_TRAIN = A_TRAIN + irregular(2:319) + trend(2:319);
FORECAST = FORECAST + irregular(320:355) + trend(320:355); 
A_FORECAST = A_FORECAST + irregular(320:355) + trend(320:355); 

%% Stage 3g: Line plot training/forecast vs original.
%figure 
%plot(2:319, [TRAIN(2:319) A_TRAIN]); 
%xlabel('Months'); 
%ylabel('Flow m^{3}/s'); 
%legend('Observed', 'ARIMA');
%xlim([0 319]);
%ylim([0 700]);

%figure 
%plot(1:36, [FORECAST(1:36) A_FORECAST]); 
%xlabel('Months'); 
%ylabel('Flow m^{3}/s'); 
%legend('Observed', 'ARIMA');
%xlim([0 36]);
%ylim([0 700]);

%% Stage 3h: Conduct performance evaluations : R2, MNS and RMSE.
%%%% TRAIN %%%%

% TRAIN: R^2 = 0.5340
%lin_mdl_TRAIN = fitlm(TRAIN(2:319), A_TRAIN);
%R2_TRAIN = lin_mdl_TRAIN.Rsquared.Ordinary; 
%xlim([0 319]);
%ylim([0 700]);

% TRAIN: Scatter plot + check with residual plot to check for bias.
%figure
%scatter(TRAIN(2:319), A_TRAIN); 
%lsline; 
%xlabel('Observed (m^{3}/s)');
%ylabel('Forecasted (m^{3}/s)'); 
%xlim([0 700]);
%ylim([0 450]);
%hold on 
%plot(xlim, ylim); 
%legend('R^{2} = 0.5340', 'y = x', 'y = 0.7284x + 41.0320');
%box on 

%figure
%plotResiduals(lin_mdl_TRAIN, 'fitted');

% TRAIN: MNS = 0.1715
%MNS_TRAIN= 1-(sum(abs(TRAIN(2:319) - A_TRAIN))/...
%    sum(abs(TRAIN(2:319)-mean(TRAIN(2:319)))));

% TRAIN: RMSE = 87.2587
%RMSE_TRAIN = sqrt(mean((TRAIN(2:319) - A_TRAIN).^2));


%%%% FORECAST %%%% 3, 6, 12, 18, 24, 36

% FORECAST: R^2 = 
lin_mdl_FORECAST = fitlm(FORECAST_RAW, A_FORECAST);
R2_FORECAST = lin_mdl_FORECAST.Rsquared.Ordinary; 
%xlim([0 36]);
%ylim([0 700]);

% FORECAST: MNS = 
MNS_FORECAST= 1-(sum(abs(FORECAST(1:36) - A_FORECAST(1:36)))/...
    sum(abs(FORECAST(1:36)-mean(FORECAST(1:36)))));

% FORECAST: RMSE =
RMSE_FORECAST = sqrt(mean((FORECAST(1:36) - A_FORECAST(1:36)).^2));

figure
subplot(3,2,1);
scatter(FORECAST(1:3), A_FORECAST(1:3)); 
lsline; 
xlabel('Observed (m^{3}/s)');
ylabel('Forecasted (m^{3}/s)'); 
xlim([0 700]);
ylim([0 600]);
hold on 
plot(xlim, ylim); 
legend('R^{2} = 0.9941', 'y = 1.1061x + 35.1380',  'y = x');
box on 

subplot(3,2,2);
scatter(FORECAST(1:6), A_FORECAST(1:6)); 
lsline; 
xlabel('Observed (m^{3}/s)');
ylabel('Forecasted (m^{3}/s)'); 
xlim([0 700]);
ylim([0 600]);
hold on 
plot(xlim, ylim); 
legend('R^{2} = 0.9790', 'y = 0.9963x + 48.6100',  'y = x');
box on 

subplot(3,2,3);
scatter(FORECAST(1:12), A_FORECAST(1:12)); 
lsline; 
xlabel('Observed (m^{3}/s)');
ylabel('Forecasted (m^{3}/s)'); 
xlim([0 700]);
ylim([0 600]);
hold on 
plot(xlim, ylim); 
legend('R^{2} = 0.9220','y = 1.1430x + -22.1280', 'y = x');
box on 

subplot(3,2,4);
scatter(FORECAST(1:18), A_FORECAST(1:18)); 
lsline; 
xlabel('Observed (m^{3}/s)');
ylabel('Forecasted (m^{3}/s)'); 
xlim([0 700]);
ylim([0 600]);
hold on 
plot(xlim, ylim); 
legend('R^{2} = 0.8876', 'y = 1.0876x + 3.5806',  'y = x');
box on 

subplot(3,2,5);
scatter(FORECAST(1:24), A_FORECAST(1:24)); 
lsline; 
xlabel('Observed (m^{3}/s)');
ylabel('Forecasted (m^{3}/s)'); 
xlim([0 700]);
ylim([0 600]);
hold on 
plot(xlim, ylim); 
legend('R^{2} = 0.8674', 'y = 1.0469x + -7.3006',  'y = x');
box on 

subplot(3,2,6);
scatter(FORECAST(1:36), A_FORECAST(1:36)); 
lsline; 
xlabel('Observed (m^{3}/s)');
ylabel('Forecasted (m^{3}/s)'); 
xlim([0 700]);
ylim([0 600]);
hold on 
plot(xlim, ylim); 
legend('R^{2} = 0.9246', 'y = 1.1569x + -28.7220',  'y = x');
box on 

%figure
%plotResiduals(lin_mdl_FORECAST, 'fitted');
%Visually the residuals should be scattered about the zero: 
%https://www.quora.com/Are-high-R-squared-values-always-great


%%%%%%%%%%%%%%%%%% END OF ARIMA MODEL DEVELOPMENT %%%%%%%%%%%%%%%%%%


%% Stage 5a: LSTM
 















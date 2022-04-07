%% Stage 0: Time series characteristics of daily precipitation & flow 

% Plot of precipitation and flow on the same graph.
table_kegworth_flow = array2table(kegworth_flow); 

time = datetime(table_kegworth_flow.kegworth_flow3,...
    table_kegworth_flow.kegworth_flow2,...
    table_kegworth_flow.kegworth_flow1);

figure
plot(time, table_kegworth_flow.kegworth_flow4);
xlabel('Time (Days)');
ylabel('Flow (m^{3}s^{-1})'); 
hold on 
yyaxis right
plot(time, kegworth_precip.Precipitation);
set(gca, 'YDir','reverse');
ylabel('Precipitation (mm)'); 
ylim([0 200]); 
legend('Flow', 'Precipitation')

% Conduct Ljung-Box test to check for autocorrelation for flow...
% @ alpha = 0.05.
res_flow = table_kegworth_flow.kegworth_flow4 -...
    mean(table_kegworth_flow.kegworth_flow1);
[h1_flow, pValue_flow, stat_flow, cValue_flow] =...
    lbqtest(res_flow, 'Lags', 20); 
disp('Daily flow series test for autocorrelation:')
disp(h1_flow); 
disp(pValue_flow); 
disp(stat_flow); 
disp(cValue_flow); 

% Conduct Ljung-Box test to check for autocorrelation for...
% precipitation @ alpha = 0.05.

res_precip = kegworth_precip.Precipitation -...
    mean(kegworth_precip.Precipitation);
[h1_precip, pValue_precip, stat_precip, cValue_precip] =...
    lbqtest(res_precip, 'Lags', 20);
disp('Daily precipitation series test for autocorrelation:')
disp(h1_precip); 
disp(pValue_precip); 
disp(stat_precip); 
disp(cValue_precip); 

% Call the modified Mann-Kendall test for flow (mmtest).
[H, pp_value, test_stat, crit] = mmtest(res_flow, 0.05); 
disp('Modified Mann-Kendall test results for flow:')
disp(H);
disp(pp_value);
disp(abs(test_stat)); 
disp(crit); 

% Call the modified Mann-Kendall test for flow (mmtest).
[H_2, pp_value_2, test_stat_2, crit_2] = mmtest(res_precip, 0.05); 
disp('Modified Mann-Kendall test results for precipitation:')
disp(H_2);
disp(pp_value_2);
disp(abs(test_stat_2)); 
disp(crit_2); 

% Compute Pearson's correlation coefficients between flow and...
% precipitation
rho = corrcoef(table_kegworth_flow.kegworth_flow4,...
    kegworth_precip.Precipitation);
disp('PMCC:')
disp(rho); % Low value 0.3315 indicates weak correlation, therefore...
% precipitation was not considered at a unique input variable +...
% the results of the MK test show neither exhibit a trend over time +...
% precipitation more strongly than flow. 

% Kendall's rank correlation between flow and precipitation.
tau = corr(res_flow, res_precip,'type', 'Kendall'); 
disp('Kendalls rank correlation coefficient:'); 
disp(tau);

%% Stage 1: Re-format original Kegworth flow series.

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
plot(months, tmaxVals.GroupMax); 
xlabel('Months');
ylabel('Flow (m^{3}s^{-1})');
xlim([0 355]);

%% Stage 2a: Decompose time series using SSA.

[trend, seasonal, irregular] = trenddecomp(tmaxVals.GroupMax, ...
    "ssa", (355-1)/2, NumSeasonal=1);
% The lag value must be a positive real scalar between 3 and N/2...
% where N is the number of elements in the first input.
% Therefore, 355 - 1/2 =  N 

% Long-term trend component + irregular component: 
figure 
subplot(4,1,1);
plot(months, irregular,'black'); 
xlabel('Months');
ylabel('I_t');
xlim([0 355]);
hold on 
plot(1:355, trend,'--');

% Seasonal component (used to support ACF plot results and model):
subplot(4, 1, 2);
plot(1:355, seasonal,'black'); 
xlabel('Months');
ylabel('S_t');
xlim([0 355]);

%% Stage 2b: Determine optimal lag number via BIC methodology.

% (1) Create zeros matrix to store all 16 combinations of ARIMA models...
LogL = zeros(4,4); 
PQ = zeros(4,4);
for p = 1:4
    for q = 1:4
        Mdl = arima(p, 0, q); % d = 0, since time series will not...
        % undergo time series transformation via differencing.
        [EstMdl,~,LogL(p, q)] = estimate(Mdl, tmaxVals.GroupMax, ...
            'Display', 'off');
        PQ(p, q) = p + q;
     end
end

% (2) Calculate BIC per combination: 
logL = LogL(:);
pq = PQ(:);
[~,bic] = aicbic(logL, pq + 1, 355);
BIC = reshape(bic, 4, 4); 

% (3) Obtain the optimal number of lags by finding minimum BIC value:
minBIC = min(BIC,[],'all');
minP = find(minBIC == BIC); 

%% Stage 2c: Check estimated optimal model with ACF and PACF + seasonality.
% Interpretation: https://www.baeldung.com/cs/acf-pacf-plots-arma-modeling

% Calculate ACF and PACF values to 50 lags.
lags = (0:1:50); 
auto = autocorr(tmaxVals.GroupMax, 'NumLags', length(lags)-1); 
parauto = parcorr(tmaxVals.GroupMax, 'NumLags', length(lags)-1);

% Set bounds at 95% confidence interval.
upper_bound = 0.05 + zeros(1, 51); 
lower_bound = -0.05 + zeros(1, 51); 

% 1st plot: ACF w/ bounds.
subplot(4, 1, 3) 
stem(lags, auto, 'black'); 
xlabel('lag {\tau}'); 
ylabel('R ({\tau})');
hold on 
plot(lags, upper_bound, '--'); % Plot upper bound
hold on
plot(lags, lower_bound, '--'); % Plot lower bound 
xlim([0 50]);

% 2nd plot: PACF w/ bounds. 
subplot(4, 1, 4)
stem(lags, parauto, 'black'); 
xlabel('lag {\tau}'); 
ylabel('R_{partial} ({\tau})');
hold on 
plot(lags, upper_bound,  '--'); % Plot upper bound
hold on
plot(lags, lower_bound,  '--'); % Plot lower bound 
xlim([0 50]);

%% Stage 2d: Conduct Leybourne-McCabe test for stationarity.

% (4) Input 'minP' into modified LM test to check for stationarity
% If h = 0 then accept the null hypothesis. If h = 1 reject null.
[h, p_value, ~, ~] = lmctest(tmaxVals.GroupMax,...
    "trend", false, "Lags", 4, "Test", "var2", "alpha", 0.05);
disp(h);
disp(p_value);
% In this case h = 0 therefore time series is stationary.
% no additional stationary pre-processing is required.

%% Stage 3: Normalise time series & split ARIMA time series.
%muu = mean(tmaxVals.GroupMax); 
%sd =  std(tmaxVals.GroupMax);

% Split to training & testing (forecasting period).
% Training normalisation.
training_start = tmaxVals.GroupMax(1:319); 
testing_start = tmaxVals.GroupMax(320:355); 

muu = mean(training_start);
sd = std(training_start); 

training_norm = tanh_norm(training_start, muu, sd); 
testing_norm = tanh_norm(testing_start, muu, sd); 

%training_norm = norm_ts(1:319);
%testing_norm = norm_ts(320:355);

% Create ARIMA model w/ t-distribution. 
arima_model = arima(4,0,4); 
arima_model.Distribution = 't'; 

% Estimate model with normalised training data 
est_arima_model = estimate(arima_model, training_norm, 'Display', 'off');
residuals1 = infer(est_arima_model, training_norm);
y_norm_train = training_norm - residuals1;

% Estimate test values with estimated arima
residuals2 = infer(est_arima_model, testing_norm); 
y_norm_test = testing_norm - residuals2;

% Converts normalised arima generated data to original scale using training
% data mean and standard deviation. 
training_org = (sd*(atanh((2.*y_norm_train) - 1))./0.01) + muu;
testing_org = (sd*(atanh((2.*y_norm_test) - 1))./0.01) + muu;

% Line plot of ARIMA and original time series against time for training.
figure
plot(1:355, tmaxVals.GroupMax); 
hold on
plot(1:319, training_org);
hold on 
plot(320:355, testing_org, 'Color', '#EDB120');
xlabel('Months'); 
ylabel('Flow (m^{3}s^{-1})'); xlabel('Months'); 
ylabel('Flow (m^{3}s^{-1})'); 
xlim([0 355]); 
ylim([0 700]);
hold on
plot(1:355, zeros(1, 355) + mean(tmaxVals.GroupMax(1:319)), '--');
hold on 
plot(1:355, zeros(1, 355) + mean(tmaxVals.GroupMax(1:319)) +...
    1.5*std(tmaxVals.GroupMax(1:319)), '--');
hold on 
hp = patch([320 354 354 320], [1 1 700 700], [245/255, 245/255, 245/255],...
    'LineStyle', 'none'); 
uistack(hp, 'bottom');
legend('Forecasted', 'Observed', 'ARIMA: Training', 'ARIMA: Validation',...
    'Mean', '1.5 Standard Deviation above Mean');

% Scatter plot training 
figure
scatter(tmaxVals.GroupMax(1:319), training_org);
lsline;
legend('y = 0.3662 + 78.2894', 'R^{2} = 0.3459');
xlabel('Observed (m^{3}s^{-1})');
ylabel('Forecasted (m^{3}s^{-1})'); 

% Scatter plot testing 
figure
scatter(tmaxVals.GroupMax(320:355), testing_org);
lsline;
legend('y = 0.3651 + 80.5492', 'R^{2} = 0.2888');
xlabel('Observed (m^{3}s^{-1})');
ylabel('Forecasted (m^{3}s^{-1})'); 
xlim([0 600]);
ylim([0 350]);

% Calculate R^2 (coeff of determination) for training and testing.
lin_mdl_train = fitlm(tmaxVals.GroupMax(1:319), training_org);
RSQ_train = lin_mdl_train.Rsquared.Ordinary; 
lin_mdl_test = fitlm(tmaxVals.GroupMax(320:355), testing_org);
RSQ_test = lin_mdl_test.Rsquared.Ordinary; 

% Calculate RMSE for training and testing.
RMSE_train = sqrt(mean((tmaxVals.GroupMax(1:319) - training_org).^2));
RMSE_test = sqrt(mean((tmaxVals.GroupMax(320:355) - testing_org).^2));

% Calculate modified NSE for training and testing.
MNS_train = 1-(sum(abs(tmaxVals.GroupMax(1:319)-training_org))/...
    sum(abs(tmaxVals.GroupMax(1:319) -...
    mean(tmaxVals.GroupMax(1:319)))));
MNS_test = 1-(sum(abs(tmaxVals.GroupMax(320:355)-testing_org))/...
    sum(abs(tmaxVals.GroupMax(320:355) -...
    mean(tmaxVals.GroupMax(320:355)))));

%% Stage 4: 
XTrain = num2cell(training_norm(1:end-1, :)); % training_norm
TTrain = num2cell(training_norm(2:end, :)); 
XTest = num2cell(testing_norm(1:end-1, :)); 
TTest = num2cell(testing_norm(2:end, :)); 

% Define LSTM network architecture
featureDimension = size(XTrain{1}, 1);
numResponses = size(TTrain{1}, 1);
numHiddenUnits = 52; % Amount of information remembered between time steps
LSTMDepth = 2;
layers = sequenceInputLayer(featureDimension);

for i = 1:LSTMDepth
    layers = [layers;lstmLayer(numHiddenUnits,OutputMode="sequence")];
end

layers = [layers
    fullyConnectedLayer(numResponses)
    regressionLayer];

% Specify training hyperparameters
options = trainingOptions("adam", ...
    ExecutionEnvironment="auto", ...
    MaxEpochs=104, ...
    MiniBatchSize=18, ...
    ValidationData={XTest, TTest}, ...
    InitialLearnRate=0.0001, ...
    LearnRateDropFactor=0.3303, ... 
    LearnRateDropPeriod=15, ...
    GradientThreshold=1, ...
    Shuffle="never", ... 
    Verbose=1);

% Train LSTM network
net2 = trainNetwork(XTrain, TTrain, layers, options);


XTrain = num2cell(training_norm(1:end-1, :)); 
TTrain = num2cell(training_norm(2:end, :)); 
XTest = num2cell(testing_norm(1:end-1, :)); 
TTest = num2cell(testing_norm(2:end, :)); 

% Forecast (Test) LSTM network 
net2 = resetState(net2); 
[updatedLSTM, YTest] = predictAndUpdateState(net2, TTest); 

% Reverse normalisatoion
YTest = cell2mat(YTest);
TTest = cell2mat(TTest); 
%YTest = inv_tanh_norm(YTest, muu, sd);
%TTest = inv_tanh_norm(TTest, muu, sd); 

figure
plot(1:35, [TTest YTest]); 
ylim([0 700]);










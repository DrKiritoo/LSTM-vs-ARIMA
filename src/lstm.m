%% 2 Preliminary checks & Plots

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

% Conduct DW test to check for autocorrelation.
days = transpose(1:9803);
date_flow_table = table(days, table_kegworth_flow.kegworth_flow4);
mdl = fitlm(date_flow_table);
[p_value,DW] = dwtest(mdl,'approximate','both');

% Conduct seasonal Mann Kendall test to check hydrological significance.
 %[~, ~, h, sig, ~, ~, ~, ~, ~, ~, ~, ~, ~, ~, ~] =...
  %   ktaub([kegworth_precip.Precipitation...
  %   table_kegworth_flow.kegworth_flow4] , 0.1, 1); 
%test = [kegworth_precip.Precipitation...
 %    table_kegworth_flow.kegworth_flow4];
%disp(h);
%disp(sig); 

%% Stage 0: Re-format original Kegworth time series.

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

% Plot maximum 7 day total flow against months forming new time series.
figure
% subplot(4,2,1); 
plot(months, tmaxVals.GroupMax, 'black'); 
xlabel('Months');
ylabel('Q (m^3/s)');
xlim([0 355]);

%% Stage 1: Decompose time series into deterministic components.

[trend, seasonal, irregular] = trenddecomp(tmaxVals.GroupMax, ...
    NumSeasonal=1); % 2 seasonal period specified to see...
% short-term and long-term seasonality.

%% Moving average of 12 months-TEST PLOT. 

% Create a 12 month moving average trend - TEST
figure
movave = transpose(movmean(tmaxVals.GroupMax, [0 12])); 
plot(months, movave); 
xlabel('Months');
ylabel('12-month trend + noise');
xlim([0 355]);
hold on 
plot(months, irregular,'black'); 
%% Plot decomposed elements 'trenddecomp'.

% Long-term trend component + irregular component: 
figure 
subplot(4,1,1);
plot(months, irregular,'black'); 
xlabel('Months');
ylabel('I_t');
xlim([0 355]);
% Plot irregular component on the same graph as trend: 
hold on 
plot(1:355, trend,'--');

% Seasonal component:
subplot(4, 1, 2);
plot(1:355, seasonal,'black'); 
xlabel('Months');
ylabel('S_t');
xlim([0 355]);


%% Determine optimal lag number via BIC methodology.

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


%% Check estimated optimal model with ACF and PACF + seasonality.

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

%% Conduct Leybourne-McCabe test for stationarity.

% (4) Input 'minP' into modified LM test to check for stationarity
% If h = 0 then accept the null hypothesis. If h = 1 reject null.
h = lmctest(tmaxVals.GroupMax,...
    "trend", false, "Lags", 3, "Test", "var2", "alpha", 0.05);
disp(h); 
% In this case h = 0 therefore time series is stationary.
% no additional stationary pre-processing is required.

%% Stage 2a: Pre-process data for LSTM-RNN 

%% Stage 2b: Pre-process data for ARIMA
% Do (+PACF?)
% When in doubt - Fetch -> Pull -> Push 



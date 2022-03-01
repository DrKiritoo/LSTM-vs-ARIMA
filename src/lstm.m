%% Stage 0: Re-format original Kegworth time series 

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

% Plot maximum 7 day total flow against months forming new time series.
figure
subplot(4,1,1); 
plot(1:355, tmaxVals.GroupMax, 'black'); 
xlabel('Months');
ylabel('Q (m^3/s)');
xlim([0 355]);

%% Stage 1: Conduct Leybourne-McCabe test for stationarity 

[trend, seasonal, irregular] = trenddecomp(tmaxVals.GroupMax, ...
    NumSeasonal=2); % 2 seasonal period specified to see...
% short-term and long-term seasonality.
% Plot long-term trend component: 
subplot(4,1,2);
plot(1:355, trend,'black');
xlabel('Months');
ylabel('T_t');
xlim([0 355]);

% Plot seasonal component:
subplot(4, 1, 3);
plot(1:355, seasonal,'black'); 
xlabel('Months');
ylabel('S_t');
xlim([0 355]);

% Plot irregular component: 
subplot(4, 1, 4);
plot(1:355, irregular,'black'); 
xlabel('Months');
ylabel('I_t');
xlim([0 355]);

% Determine optimal lag number via BIC methodology.
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

% (3.1) Check estimated optimal model with ACF and PACF. 
auto = autocorr(tmaxVals.GroupMax); 
parauto = parcorr(tmaxVals.GroupMax);
lags = (0:1:20); 

figure 
subplot(2, 1, 1); 
plot(lags, auto, 'black'); 
xlabel('lag {\tau}'); 
ylabel('R ({\tau})');
xlim([0 20]);

subplot(2, 1, 2);
plot(lags, parauto, 'black'); 
xlabel('lag {\tau}'); 
ylabel('R_{partial} ({\tau})');
xlim([0 20]); 

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



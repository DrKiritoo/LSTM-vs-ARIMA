%% Stage 0: Re-format Kegworth time series 

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
subplot(3,1,1); 
plot(1:355, tmaxVals.GroupMax); 
xlabel('Months');
ylabel('Q* (m^3/s)');
xlim([0 355]);

%% Stage 1: Conduct test for stationarity 

% Determine trend: Plot long-term trend on graph (Using trenddecomp MATLAB)
trend = trenddecomp(tmaxVals.GroupMax); 
subplot(3,1,2);
plot(1:355, trend);
xlabel('Months');
ylabel('Q* (m^3/s)');
xlim([0 355]);

% Determine optimal lag number via BIC methodology.


% Input final parameters into modified LM test. 



%% Stage 2a: Pre-process data for LSTM-RNN 

%% Stage 2b: Pre-process data for ARIMA
% Do (+PACF?)





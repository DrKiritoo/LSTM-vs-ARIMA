% Create time series of monthy peak flow from daily flow data.
tmaxVals=rowfun(@max,kegworth,'InputVariables','Discharge', ...
                          'GroupingVariables',{'Year','Month'}, ...
                          'OutputVariableNames',{'GroupMax', 'Day'});
tmaxVals.Day(:, 1) = 1; % Ensure time-series is evenly spaced.
num_date = datetime(tmaxVals.Year, tmaxVals.Month, tmaxVals.Day, 'Format', 'MMM-yyyy');

% Plot date of max vs max monthly discharge.
figure
plot(num_date, tmaxVals.GroupMax); 
xlabel('Month')
ylabel('Peak Flow (m^3/s)')

% Check if data has missing values.
max_flows = tmaxVals.GroupMax; 
len = length(max_flows(:,1));
missing_values = 0; 

for i = 1:len
    if isnan(max_flows(i, 1))
        missing_values = missing_values + 1; 
        max_flows(i, 1) = 0;
    end
end

% Percentage of data missing.
disp(missing_values * 100 / len);



        









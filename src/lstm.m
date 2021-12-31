% Create time series of monthy peak flow from daily flow data.
tmaxVals=rowfun(@max,kegworth,'InputVariables','Discharge', ...
                          'GroupingVariables',{'Year','Month'}, ...
                          'OutputVariableNames',{'GroupMax', 'Date'});

% Convert day, month, year to a single datetime variable.
num_dates = datetime([tmaxVals.Year, tmaxVals.Month, tmaxVals.Date],...
    "Format","dd-MMM-uuuu"); 
month_max = table(num_date, tmaxVals.GroupMax); 

% Plot date of max vs max monthly discharge.
figure
plot(month_max.num_date, month_max.Var2);
xlabel('Month')
ylabel('Peak Flow (m^3/s)')

% Check if data has missing values.
max_flows = table2array(month_max(:,2)); 
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



        








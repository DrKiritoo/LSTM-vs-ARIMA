% Create time series of monthy peak flow from daily flow data.
% Create 7 day rolling sum window.

new_keg = kegworth(4474:15280, :); % Start after 7 year gap.
t = datetime(1991,3,1):calmonths(1):datetime(2020,9,30);
daysPerMonth = days(diff(t)); % Calculate the number of days per month. 

window_sum = zeros(1, length(daysPerMonth));

for j = 1:daysPerMonth(1,:)
    % Get 7 day window sum of discharges per month. 
    window_sum(1, j) = sum(new_keg.Discharge(j:j+6, 1)); 
end












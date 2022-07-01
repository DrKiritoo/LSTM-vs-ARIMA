function [x] = inv_min_max(xnorm, original)
x = (xnorm.*(max(original) - min(original))) + min(original);
end


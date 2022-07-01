function [x_norm] = min_max(x, original)
x_norm = (x - min(original))/(max(original)-min(original)); 
end


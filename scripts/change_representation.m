function outSet = change_representation(set,segNum)
% This function have to be called after adder to extend time-series of a testing set.
% It divides the set into segments of segNum fragments.
% It records each fragment as an average (feature by feature).

% set - set to extend
% segNum - number of segments

if nargin < 2
    segNum = 10;
end

outSet = cell(1,length(set));
for i=1:length(set) 
	temp = set{i}';

	data_len = size(temp,1);
	data_len = ceil(data_len / segNum) * segNum;
	newLen = data_len;
	
    if size(temp,1) >= 3
	    [initSize1, initSize2] = ndgrid(1:size(temp, 1), 1:size(temp, 2));
	    [newSize1, newSize2] = ndgrid(linspace(1, size(temp, 1), newLen), 1:size(temp, 2));
	    newData = interpn(initSize1, initSize2, temp, newSize1, newSize2);
    else
        newData = temp;
    end

	segSize = floor(data_len/segNum);
	 
	representation = [];
    for d=1:size(newData,2)
		data = newData(:,d);	 
		data = (data - mean(data))/std(data);
		dane=[reshape(data,segSize,[])]; 
        dane(isnan(dane)) = 0;		    
		segments = mean(dane);    
		
		TF = isnan(segments);
		if TF		 
		    segments(isnan(segments))=0;		 
		end		
		representation = [representation;segments];
    end
    outSet{i} = representation;
end
 
 
 
 
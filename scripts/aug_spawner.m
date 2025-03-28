% SPAWNER - Suboptimal Warped Time Series Generator

% K. Kamycki, T. Kapuścinski, M. Oszust, Data augmentation with sub-
% optimal warping for time-series classification, Sensors 20 (1) (2020).

function [outTrain, outTrainLabels] = aug_spawner(train,trainLabels,nDraws)
% train - multivariate training set of time-series
% trainLabels - training set labels
% nDaws - number of augmentation epochs

outTrain = cell(1,length(trainLabels)*nDraws);
outTrainLabels = zeros(1,length(trainLabels)*nDraws);
xSize=size(train,2);
arrPod=zeros([xSize, xSize]);

for i=1:1:xSize  
	tempI = train{i};
	for j=i+1:1:xSize  
		tempJ = train{j};
 
		window=ceil(size(tempI,2)/10);			 
		if size(tempI,2)< size(tempJ,2)
			window=ceil(size(tempJ,2)/10); 
		end			

		try 
			dt = dtw(tempI,tempJ,window)/(size(tempI,2)+size(tempJ,2));
			if isnan(dt)
				dt = dtw(tempI,tempJ)/(size(tempI,2)+size(tempJ,2));
			end
		catch ME
			dt=10e10;
		end
		 
		arrPod(i,j) = dt; 
		arrPod(j,i) = dt;
	end %j 
end %i

arrAvgClass=[];	
for i=1:xSize 
	avgClass=0;
	tmpCount=0;
	for j=1:xSize 		
		if trainLabels(i) == trainLabels(j)
			avgClass=avgClass+arrPod(i,j);
			tmpCount=tmpCount+1;
		end
	end
	avgClass=avgClass/(tmpCount-1);
	arrAvgClass=[arrAvgClass;avgClass];
end

arrAvgClassMin=[];	
for i=1:xSize 
	minimum=1000e6;			 
	for j=1:xSize 		
		if trainLabels(i) == trainLabels(j)
			if i~=j					
				if minimum > arrPod(i,j)
					minimum = arrPod(i,j);
				end
			end
		end
	end
	arrAvgClassMin=[arrAvgClassMin;minimum]; 			
end

arrClass=[];
for n=1:max(trainLabels)
	avgClass=0;
	tmpCount=0;
	for i=1:xSize 
		for j=i+1:xSize 		
			if trainLabels(j) == (n)  
				avgClass=avgClass+arrPod(i,j);
				tmpCount=tmpCount+1;
			end
		end
	end
	avgClass=avgClass/(tmpCount-1);
	arrClass=[arrClass;avgClass];
end

jump=1;
checked=zeros([xSize,xSize]);
counter=1;
for i=1:jump:xSize   
	tempI = train{i};  
	odl4I=arrPod(i,:);		

	[~,ind] = sort(odl4I,'asc');

	ind=ind(2:length(ind));  
	tmpCount=0;
	for j=ind(1:length(ind) ) 
		tempJ = train{j};
		if trainLabels(i) == trainLabels(j)   
			if checked(i,j)<=0
				checked(i,j)=checked(i,j)+1;
				checked(j,i)=checked(j,i)+1;
				for draw=1:nDraws
					randAB=rand();
					drawA=size(tempI,2)*randAB;
					drawB=size(tempJ,2)*randAB;
					drawA=ceil(drawA);
					drawB=ceil(drawB);

					t1a=tempI(:, 1:drawA); 
					t1b=tempI(:, drawA+1: size(tempI,2));     
					t2a=tempJ(:, 1:drawB); 
					t2b=tempJ(:, drawB+1: size(tempJ,2)); 							  

					ct1a=ceil(size(t1a,2)/10);
					ct2a=ceil(size(t2a,2)/10);
					ct1b=ceil(size(t1b,2)/10);
					ct2b=ceil(size(t2b,2)/10);
							  
					if ct1a>ct2a  
						window = ct1a;
					else 
						window = ct2a;
					end
					if window < 1
						window = 1;
					end

					try 
						[~,ix1,iy1] = dtw(t1a,t2a,window);
					catch ME
						continue
					end
					
					if size(t1a,2)>1 | size(t2a,2)>1 
						onewarp1 = t1a(:,ix1);
						twowarp1 = t2a(:,iy1);	
					else
						onewarp1 = t1a;
						twowarp1 = t2a;
					end

					if ct1b > ct2b  
						window = ct1b;
					else 
						window = ct2b;
					end
					if window < 1
						window = 1;
					end
					[~,ix2,iy2] = dtw(t1b,t2b,window);
					if size(t1b,2)>1 | size(t2b,2)>1
						onewarp2 = t1b(:,ix2);
						twowarp2 = t2b(:,iy2);	
					else
						onewarp2 = t1b;
						twowarp2 = t2b;
					end
   
					together1 = normrnd(0.5*(onewarp1+twowarp1),0.05*abs(onewarp1-twowarp1)); 
					together2 = normrnd(0.5*(onewarp2+twowarp2),0.05*abs(onewarp2-twowarp2));
					together = [together1,together2];  

					outTrainLabels(counter)=trainLabels(i);	
					outTrain{counter}=together;		 
					counter=counter+1;	
					tmpCount=tmpCount+1;	
				end
			end
		end
	end %j
end %i
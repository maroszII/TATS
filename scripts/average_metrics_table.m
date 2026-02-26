% average_metrics_table averages numerical metrics across multiple tables.
%
%   Input:
%   - allTables: A cell array (e.g., 10x1) where each cell contains 
%                a table returned by calculateMetrics.
%
%   Output:
%   - averagedTable: A single table with averaged Precision, Recall, 
%                    F1_Score, and Accuracy.

function averagedTable = average_metrics_table(allTables)
    % Initialization
    numIterations = length(allTables);
    if numIterations == 0
        error('The input cell array is empty.');
    end

    % Get metadata from the first table (labels and headers)
    templateTable = allTables{1};
    classNames = templateTable.Class;
    varNames = templateTable.Properties.VariableNames;
    
    % Get dimensions (rows and numeric columns)
    [numRows, numCols] = size(templateTable);
    
    % Extract and sum numeric data
    % Columns 2 to end are Precision, Recall, F1_Score, and Accuracy
    numericSum = zeros(numRows, numCols - 1);
    
    for i = 1:numIterations
        % Convert table numeric columns to array
        currentNumericData = table2array(allTables{i}(:, 2:end));
        
        % Add to sum
        numericSum = numericSum + currentNumericData;
    end
    
    % Calculate the mean
    numericAverage = numericSum / numIterations;
    
    % Reconstruct the final table
    % We concatenate the original Class column with the averaged numeric data
    averagedTable = table(classNames, ...
                          numericAverage(:,1), ...
                          numericAverage(:,2), ...
                          numericAverage(:,3), ...
                          'VariableNames', varNames);
end
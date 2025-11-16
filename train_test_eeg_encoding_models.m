% ========================================================================
% ASSIGNMENT 5: EEG encoding models
% Sopie Charlotte Masak
% ========================================================================

%% part I:  effect of training data amount on encoding accuracy

%%% train the encoding models

% this part of the script executes an experiment to systematically 
% determine the relationship between the amount of training data and the 
% prediction accuracy of the EEG encoding model.

% define the training data sizes to test
trainSizes = [250, 1000, 10000, 16540];

% initiate a cell array to store the mean correlation results (meanR) for each size
all_meanR = cell(1, length(trainSizes));

% Load the data
load("data_assignment_5.mat")

% Get the data dimension sizes
[numTrials, numChannels, numTime] = size(eeg_train);
numFeatures = size(dnn_train, 2);

% to evaluate the prediction accuracy of the EEG encoding model based on
% the amount of training data, we have loop the training of the model over
% the data sizes.

%%% ------------- START LOOP ------------- %%%

for k = 1:length(trainSizes)
    currentSize = trainSizes(k);

    % 1. subsample the training data
    % randomly select 'currentSize' trials for training
    % randperm(N, K) gets K unique random indices from 1 to N
    subsample_indices = randperm(numTrials, currentSize);
    
    % create the subsampled training data
    eeg_train_sub = eeg_train(subsample_indices, :, :); % [currentSize x numChannels x numTime]
    dnn_train_sub = dnn_train(subsample_indices, :);    % [currentSize x numFeatures]
    
    % initialize array for regression coefficients (W) and intercepts (b)
    % for this iterations size --> they will be calculated in training
    % process to best fit the data
    W = zeros(numFeatures, numChannels, numTime);
    b = zeros(numChannels, numTime);
    
    % progressbar parameters (using subsampled data size for counting)
    totalModels = numChannels * numTime;
    modelCount = 0;

    % 2. train the encoding models on subsampled data
    % train linear regression independently for each EEG channel and time point
    for ch = 1:numChannels
        for t = 1:numTime
            
            % extract EEG responses for this channel/time over all trials
            y = eeg_train_sub(:, ch, t); % [currentSize x 1]
            
            % fit linear regression: y = DNN * w + b
            % note: Using dnn_train_sub as the predictor
            mdl = fitlm(dnn_train_sub, y);
            
            % Ssve parameters
            W(:, ch, t) = mdl.Coefficients.Estimate(2:end); % weights
            b(ch, t)    = mdl.Coefficients.Estimate(1);     % intercept
            
            % update progress bar
            modelCount = modelCount + 1;
            if mod(modelCount, 500) == 0 || modelCount == totalModels
                fprintf('\rTraining models (Size %d): %d / %d (%.1f%%)', ...
                    currentSize, modelCount, totalModels, 100*modelCount/totalModels);
            end
        end
    end

    % 3. use trained models to predict EEG responses for the full test images

    % get number of test images
    [numTest, ~] = size(dnn_test);

    % initialize matrix of zeros for predictions with correct dimensions
    eeg_test_pred = zeros(numTest, numChannels, numTime);
    
    % loop over EEG channels and time points
    % apply linear model to all test images
    for ch = 1:numChannels
        for t = 1:numTime
            eeg_test_pred(:, ch, t) = dnn_test * W(:, ch, t) + b(ch, t);
        end
    end

    % 4. compute prediction accuracy using Pearson's correlation
    % r between real and predicted EEG for each channel and time point

    % get numbers of real EEG
    [Ntest, Nchannels, Ntime] = size(eeg_test);

    % initialize correlation matrix
    R = zeros(Nchannels, Ntime);
    
    % loop over all EEG channels and time points
    for ch = 1:Nchannels
        for t = 1:Ntime
            % extract real and predicted values across all images
            real_vec = squeeze(eeg_test(:, ch, t));
            pred_vec = squeeze(eeg_test_pred(:, ch, t));
            % compute Pearson's r
            % outcome is a channels x time matrix of predictive accuracy
            R(ch, t) = corr(real_vec, pred_vec, 'Type', 'Pearson');
        end
    end

    % 5. store the results
    % average the correlation across channels
    % restults in one correlation value per time point
    meanR = mean(R, 1);

    % store averaged correlation vector for iteration k in cell array
    all_meanR{k} = meanR;
    
end

%%% ------------- END LOOP ------------- %%%


%%% plot prediction accuracies 

% this part of the code visualizes how the amount of training data affects
% the encoding accuracy of the EEG model over time. each curve corresponds
% to one value in trainSizes. the x-axis represents time in ms and the
% y-axis shows the mean Pearson correlation between the model's predicted
% EEG response and the real EEG data.

% open new plotting window
figure;

% tell MATLAB we wanna draw multiple curves in same plot
hold on;

% get distinct colors for the lines
colors = lines(length(trainSizes));

% figure out length of time axis (number of time points)
Ntime = length(all_meanR{1});

% iterate over training sizes
for k = 1:length(trainSizes)
    % take mean accuracy time course and assign color
    % use DisplayName to automatically show in legend
    plot(all_meanR{k}, 'Color', colors(k, :), ...
        'LineWidth', 2, ...
        'DisplayName', sprintf('N = %d', trainSizes(k)));
end

% done adding lines, so release the hold
hold off;

% add axis labels and formatting
xlabel('Time (ms)');
ylabel('Mean Pearson Correlation (Encoding Accuracy)');
title('Effect of Training Data Amount on Encoding Accuracy');
grid on;        % makes reading easier
legend show;    % use DisplayName from before
set(gca, 'FontSize', 14);

% label x-axis with the time values from EEG
% problem: Ntime = 50 and labeling every tick will break the plot
% solution: just label every 10th tick
idx = round(linspace(1, Ntime, 10)); % pick ~10 ticks
xticks(idx);
xticklabels(string(times(idx)));


%%% interpretation

% the plot shows a monotonic improvement with more training samples. with
% training data amount N = 10,000 and N = 16,540 almost no differences are
% visible, the curves are almost identical. this indicates that the model
% reaches a plateau/saturation after about N = 10,000 and adding more data 
% does not make a difference.

% also, the time course of the plot is meaningful. peaks around 100-130ms
% are where early visual EEG components are expected. the correlation drops
% after abut 200-350ms, but even there, more training comes with better
% prediction. 

% the negative or near-zero correlations before stimulus onset at 0ms is a
% good sign. it shows that the model predicts only stimulus-driven activity.

% ========================================================================

%% part II:  effect of DNN feature amount on encoding accuracy

% this part of the script investigates how the number of DNN features used
% as predictors affects the performance of the EEG encoding model. to do
% this, i repeat the same pipeline as before -- but instead of looping over
% trainSizes, i loop over feature amounts.

%%% train the encoding models

% this part of the script trains an encoding model for every channel and
% timepoint. it predicts EEG responses and computes Pearsons correlations
% between the predicted and real EEG. then it averages correlations across
% channels to have one time course.

% define the amounts of DNN features to test
featureSizes = [25, 50, 75, 100];

% initiate a cell array to store the results
all_meanR_features = cell(1, length(featureSizes));

% the training of the model is looped over the feature sizes.

%%% ------------- START LOOP ------------- %%%

for k = 1:length(featureSizes)
    
    F = featureSizes(k);

    % select first F columns of the DNN feature matrix
    dnn_train_sub = dnn_train(:, 1:F);
    dnn_test_sub  = dnn_test(:, 1:F);

    % initialize storage for regression coefficients (W) and intercepts (b)
    % only for the current iteration
    W = zeros(F, numChannels, numTime);
    b = zeros(numChannels, numTime);
    R = zeros(numChannels, numTime);

    % train model on subsampled data --> linear regression independently
    % for each EEG channel and time point
    for ch = 1:numChannels
        for t = 1:numTime

            % get EEG responses for this channel/time over all trials
            y = squeeze(eeg_train(:, ch, t));

            % fit linear regression
            mdl = fitlm(dnn_train_sub, y);

            % save parameters
            W(:, ch, t) = mdl.Coefficients.Estimate(2:end);
            b(ch, t)    = mdl.Coefficients.Estimate(1);
        end
    end


    % predict EEG responses on test set
    % create empty prediction array with same dimensions as eeg_test
    eeg_test_pred = zeros(size(eeg_test));
    % loop over each channel at each time point
    for ch = 1:numChannels
        for t = 1:numTime
            % apply linear model
            eeg_test_pred(:, ch, t) = dnn_test_sub * W(:, ch, t) + b(ch, t);
        end
    end

    % compute correlations per channel and time point
    for ch = 1:numChannels
        for t = 1:numTime
            % pair eeg_test reponse with eeg_test_pred response for each
            % channel x timepoint pair. squeeze() is necessary because we
            % need to remove singleton dimensions to make the sizes match,
            % otherwise corr would not work.
            R(ch, t) = corr(squeeze(eeg_test(:, ch, t)), ...
                            squeeze(eeg_test_pred(:, ch, t)));
        end
    end

    % average across channels
    all_meanR_features{k} = mean(R, 1);

end


%%% ------------- END LOOP ------------- %%%

%%% plot prediction accuracies 

% this section visualizes how the number of DNN features used for training
% affects the encoding accuracy of the EEG model over time. each curve
% corresponds to one value in featureSizes. The x-axis shows time in ms ans
% the y-axis shows the mean Pearson correlation between predicted and real
% EEG averaged across channels.

% open new figure window and allow multiple curves
figure; hold on;

% assign colors to each DNN feature sizes
colors = lines(length(featureSizes));

% loop over all feature sizes and plot their accuracy curves
for k = 1:length(featureSizes)
    plot(all_meanR_features{k}, ...
         'LineWidth', 2, ...
         'Color', colors(k,:), ...
         'DisplayName', sprintf('%d Features', featureSizes(k)));
end

% stop adding curves
hold off;

% x-axis labels and formatting
xlabel('Time (ms)');
ylabel('Mean Pearson Correlation');
title('Effect of DNN Feature Amount on Encoding Accuracy');
legend show;
grid on;


%%% interpretation

% early on, the DNN features do not encode meaningful information -- which
% is how it is supposed to be, as the stimulus onsets at 0ms.

% around 17ms, all curves spike sharply. the correlations are higher with
% more features integrated. it seems that adding more features generally
% improves encoding performance. this effect does not saturate with the
% amounts of DNN features tested here.
% =========================================================================
% ASSIGNMENT 2: WORKING WITH EEG DATA
% Sophie Charlotte Masak (5607821)
% =========================================================================

% make sure to have a clean workspace
clear

% set working directory
cd('/Users/lotte/Library/Mobile Documents/com~apple~CloudDocs/Studium/M. Sc. Cognitive Neuroscience/Intro to Programming/Assignment 2')

% load assignment data
load('eeg_data_assignment_2.mat')


% =========================================================================

%% task 2: mean eeg voltage

% assumption: column indices (1-63) correspond to ch_names vector

% get indices of all channels that *contain* the letters "O" and "F"
O = find(contains(ch_names, "O"));
F = find(contains(ch_names, "F"));

% get timepoint of 0.1 seconds
T = find(times == 0.1);

% sanity check
times(T);

% calculate the mean for occipital and frontal channels
mean_O = mean(eeg(:, O, T), "all");
mean_F = mean(eeg(:, F, T), "all");

% results
fprintf(['The mean EEG voltage at %.3f seconds for occipital channels is'...
    ' %.4f µV, for frontal channels it is %.4f µV.\n'], times(T), mean_O, mean_F);

% occipital mean: 0.4213 µV
% frontal mean: -0.0536 µV


% =========================================================================

%% task 3: timecourse of eeg voltage

figure; % open new figure window
hold on; % keep existing plots in same figure

% get mean across conditions (1) and channels (2)
% plot with time on x-axis and mean on y-axis
% mean() reduces condition dimension to 1, squeeze() removes from array
plot(times, squeeze(mean(mean(eeg, 1), 2)))

% label the axes & add title
xlabel('Time')
ylabel('EEG Voltage (µV)')
title('Average across channels and conditions')

% reset hold state
hold off


% =========================================================================

%% task 4: timecourse of channel sepcific eeg voltage

figure; % open new figure window
hold on; % keep existing plots in same figure

% get all conds (:), occipital channels (O) and all timepoints (:)
% calculate mean across all of that
% plot with time on x-axis and mean on y-axis
plot(times, squeeze(mean(eeg(:, O, :))), 'b');

% same for frontal channels, choose different colour
plot(times, squeeze(mean(eeg(:, F, :))), 'r');

% label the axes nicely & add title
xlabel('Time (s)');
ylabel('EEG Voltage (µV)');
title('Timecourse of EEG Voltage for Occipital and Frontal Channels');

% custom legend
% use dummy lines (don't show in plot because of nan) to  get colours
% necessary because we otherwise would have legend entry for each line
h1 = plot(nan, nan, 'b', 'LineWidth', 1.5);  % blue dummy
h2 = plot(nan, nan, 'r', 'LineWidth', 1.5);  % red dummy
legend([h1 h2], {'Occipital Channels', 'Frontal Channels'});

% reset hold state
hold off; 

% observations:
% The occipital lobe is the primary visual processing center of the brain. 
% The strong, time-locked electrical response in EEG suggests that the 
% person was presented with a visual stimulus. The simultaneous lack of 
% complex activation in the frontal cortex, involved in executive
% functions, indicates that the stimulus did not require a complex 
% cognitive response.


% =========================================================================

%% task 5: eeg voltage across occipital channel for 1st vs. 2nd condition

figure; % open new figure window
hold on; % keep existing plots in same figure

% select condition 1, occipital channels (O) and all timepoints (:)
% calculate mean over condition (dim 2)
plot(times, squeeze(mean(eeg(1, O, :), 2)), 'b', DisplayName = "Cond 1");

% select condition 2, occipital channels (O) and all timepoints (:)
% calculate mean over condition (dim 2)
plot(times, squeeze(mean(eeg(2, O, :), 2)), 'g', DisplayName = "Cond 2");

% label the axes & add title
xlabel('Time (s)');
ylabel('EEG Voltage (µV)');
title('Timecourse of EEG Voltage for Occipital Channels');

% show legend using DisplayName values
% works here because there's just one line per "case"
legend show;

% reset hold state
hold off; 

% observations:
% The curves are highly similar because both of them step from the
% occipital brain region. But, there were different stimuli presented! The
% most noticeable difference to me is the valley at roughly 100ms. It
% probably corresponds to N100, a negative deflection that occurs
% approximately 100ms after a stimulus. This component is sensitive to the
% nature and complexity of visual stimuli. The image 2 with the deeper
% N100 response likely received greater selective attention or required
% more cognitive resources.
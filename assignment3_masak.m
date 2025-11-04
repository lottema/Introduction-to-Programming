% =========================================================================
% ASSIGNMENT 3: COUNTERBALANCING & PSEUDORANDOMIZATION
% Sophie Charlotte Masak (5607821)

% This code creates the condition list for an experiment. Participants view
% images of faces with different emotional expressions.

% 2x3 factorial design
% familiarity = familiar vs. unfamiliar
% emotion = positive vs. neutral vs. negative
% =========================================================================

% make sure to have a clean workspace
clear

% parameters
n = 60;
familiarity = ["familiar", "unfamiliar"];
emotion = ["positive", "neutral", "negative"];

% rules!
% every trial has two words
% emotions are random
% 1/2 of participants start with familiar blocks
% 1/3 of participants start with blocks corresponding to each emotion
% blocks of same emotion are presented consecutively


% =========================================================================

% random familiarity order, each equally likely to start
% two options --> probability is 1/2
fam_order = familiarity(randperm(2))

% random emotion order, each equally likely to start
% three emotions --> probability is 1/3
emo_order = emotion(randperm(3))

% initiate object for placing block content
blocks = strings(1, 6)

% as one emotion block is always followed by the other block of this
% emotion, we can build two blocks at once.
blocks(1:2) = [emo_order(1) + "_" + fam_order(1), emo_order(1) + "_" + fam_order(2)]

% re-shuffle the familiarity and build next blocks
fam_order = familiarity(randperm(2))
blocks(3:4) = [emo_order(2) + "_" + fam_order(1), emo_order(2) + "_" + fam_order(2)]

% re-shuffle the familiarity and build next blocks
fam_order = familiarity(randperm(2))
blocks(5:6) = [emo_order(3) + "_" + fam_order(1), emo_order(3) + "_" + fam_order(2)]

% this section of code would create the conditions for one participant
% based on the given rules. as we need a condition list for n = 60, this
% needs to be put under a loop.

% =========================================================================

% initialize string matrix
all_blocks = strings(n, 6);

% for loop over 1:60 participants
for p = 1:n
    % random familiarity order (1/2 chance for either start)
    fam_order = familiarity(randperm(2));
    
    % random emotion order (1/3 chance for each start)
    emo_order = emotion(randperm(3));
    
    % create blocks container
    blocks = strings(1, 6);
    
    % build blocks for each emotion, ensuring same-emotion blocks are consecutive
    for e = 1:3
        fam_order = familiarity(randperm(2));  % reshuffle each time
        idx = (e - 1) * 2 + 1;                 % starting index for this emotion pair
        blocks(idx:idx+1) = [emo_order(e) + "_" + fam_order(1), emo_order(e) + "_" + fam_order(2)];
    end
    
    % store this participants block order
    all_blocks(p, :) = blocks;
end

% =========================================================================

% create a MATLAB table
table = array2table(all_blocks, ...
    'VariableNames', {'Block1', 'Block2', 'Block3', 'Block4', 'Block5', 'Block6'})
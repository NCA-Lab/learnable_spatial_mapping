clear;clc;close all;
rng(114514)
savepath = 'D:\EEG_dataset\dataset_KUL\kul_cv';
datasetpath = 'D:\EEG_dataset\dataset_KUL\preprocessed_data_128Hz';

% if isfolder(savepath)
%     rmdir(savepath, 's');
% end

cvfold = 5;

for subject = 14:16
    % for subject = 1

    s = load(fullfile(datasetpath, ['S', num2str(subject), '.mat']));
    data = s.preproc_trials;

    numtrial = length(data);

    for trial = 1:numtrial
        datatmp = data{trial};
        raw = datatmp.RawData;
        fs = datatmp.FileHeader.SampleRate;
        lrIdx = datatmp.attended_ear;
        direction = lrIdx;
        eeg = raw.EegData;
        leneeg = floor(length(eeg) / fs);
        lenwin = 1;
        overlap = 2;
        lenoverlap = lenwin / overlap;
        numslice = floor((leneeg / lenoverlap - (cvfold) * (overlap - 1)) / cvfold) * cvfold;
        sliceArray = 1:numslice;
        sliceArray = reshape(sliceArray, [], 5);
        sliceArray = sliceArray + repmat((0:4) * (overlap - 1), size(sliceArray, 1), 1);

        for fold = 1:cvfold
            testidx = sliceArray(:, fold);
            trainvalidx = sliceArray(:, (1:cvfold) ~= fold);
            val = randi([1, cvfold - 1]);
            validx = trainvalidx(:, val);
            trainidx = trainvalidx(:, (1:(cvfold - 1)) ~= val);
            idx = reshape(trainidx, 1, []);
            validx = reshape(validx, 1, []);
            testidx = reshape(testidx, 1, []);
            trainidx = [];
            diffidx = (diff(idx) == overlap);
            diffidx = [diffidx, 0];

            if length(idx) >= 1

                for j = 1:length(diffidx)

                    if diffidx(j)
                        trainidx = [trainidx, idx(j):(idx(j + 1) - 1)];
                    else
                        trainidx = [trainidx, idx(j)];
                    end

                end

            end

            idx = validx;
            validx = [];
            diffidx = (diff(idx) == overlap);
            diffidx = [diffidx, 0];

            if length(idx) >= 1

                for j = 1:length(diffidx)

                    if diffidx(j)
                        validx = [validx, idx(j):(idx(j + 1) - 1)];
                    else
                        validx = [validx, idx(j)];
                    end

                end

            end

            idx = testidx;
            testidx = [];
            diffidx = (diff(idx) == overlap);
            diffidx = [diffidx, 0];

            if length(idx) >= 1

                for j = 1:length(diffidx)

                    if diffidx(j)
                        testidx = [testidx, idx(j):(idx(j + 1) - 1)];
                    else
                        testidx = [testidx, idx(j)];
                    end

                end

            end

            for slice = trainidx
                eegslice = eeg((slice - 1) * fs * lenoverlap + (1:lenwin * fs), :);
                SAVEPATH = fullfile(savepath, sprintf('CV_%02d', fold), sprintf('decision_win_%02.1fs', lenwin), 'train', direction);

                if ~isfolder(SAVEPATH)
                    mkdir(SAVEPATH);
                end

                save(fullfile(SAVEPATH, sprintf('S%02d_trial_%02d_slice_%02d_way_%s', subject, trial, slice, direction)), 'eegslice');
            end

            for slice = validx
                eegslice = eeg((slice - 1) * fs * lenoverlap + (1:lenwin * fs), :);
                SAVEPATH = fullfile(savepath, sprintf('CV_%02d', fold), sprintf('decision_win_%02.1fs', lenwin), 'val', direction);

                if ~isfolder(SAVEPATH)
                    mkdir(SAVEPATH);
                end

                save(fullfile(SAVEPATH, sprintf('S%02d_trial_%02d_slice_%02d_way_%s', subject, trial, slice, direction)), 'eegslice');
            end

            for slice = testidx
                eegslice = eeg((slice - 1) * fs * lenoverlap + (1:lenwin * fs), :);
                SAVEPATH = fullfile(savepath, sprintf('CV_%02d', fold), sprintf('decision_win_%02.1fs', lenwin), 'test', direction);

                if ~isfolder(SAVEPATH)
                    mkdir(SAVEPATH);
                end

                save(fullfile(SAVEPATH, sprintf('S%02d_trial_%02d_slice_%02d_way_%s', subject, trial, slice, direction)), 'eegslice');
            end

            fprintf("Subject %02d trial %02d fold %02d complete\n", subject, trial, fold)

        end

    end

end

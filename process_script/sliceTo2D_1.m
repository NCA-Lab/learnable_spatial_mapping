clear;clc;close all;
rng(114514)
window_length = [2,5];
for lenwin = window_length
	for way = ["twoway","threeway","fourway","eightway"]
		fprintf("Starting preprocess for %s\n",way);
		savepath = "E:\EEG_dataset\classify\" + way;
		datasetpath = 'C:\Users\sean\Documents\ZYMdeDocument\21-04-EEG\Datasets\NJUNCA_preprocessed_arte_removed';
		
		% if isfolder(savepath)
		%     rmdir(savepath, 's')
		% end
		
		totalSlice = 0;
		cvfold = 5;
		subjectID = [2, 3, 4, 6, 7, 8, 9, 12, 13, 14, 15, 16, 17, 18, 19, 21, 22, 23, 24, 25, 26, 27];
		for ss = 1:length(subjectID)
			subject = subjectID(ss);
			s = load(fullfile(datasetpath, sprintf('S%02d.mat', subject)));
			data = s.data;
			expinfo = s.expinfo;
			
			numtrial = length(data.eeg);
			
			nextdirection = [];
			
			% steady
			if subject <= 20
				trialnum = 24;
			else
				trialnum = length(data.eeg);
			end
			
			for trial = 1:trialnum
				% for trial = 30
				lrIdx = expinfo.attended_lr(trial) == "right";
				lrIdx = lrIdx + 1;
				
				if subject <= 20
					azimuth = expinfo.azimuth{trial}{1};
				else
					azimuth = expinfo.azimuth{trial};
				end
				
				azimuth = sort(azimuth, 'ascend');
				assert(length(azimuth) == 2)
				azimuth = azimuth(lrIdx);
				
				% twoway
				if way == "twoway"
					if azimuth < 0
						direction = 'L';
					elseif azimuth > 0
						direction = 'R';
					else
						
						if isempty(nextdirection)
							i = rand(1);
							i = round(i);
							
							if i == 0
								direction = 'L';
								nextdirection = 'R';
							elseif i == 1
								direction = 'R';
								nextdirection = 'L';
							end
							
						else
							direction = nextdirection;
							nextdirection = [];
						end
						
					end
					
					% threeway
				elseif way == "threeway"
					if azimuth < -30
						direction = "L";
					elseif azimuth > 30
						direction = "R";
					else
						direction = "C";
					end
					
					% fourway
				elseif way == "fourway"
					if azimuth < -45
						direction = "L";
					elseif (azimuth < 0) && (azimuth >= -45)
						direction = "LC";
					elseif (azimuth > 0) && (azimuth <= 45)
						direction = "RC";
					elseif (azimuth > 45)
						direction = "R";
					else
						
						if isempty(nextdirection)
							i = rand(1);
							i = round(i);
							
							if i == 0
								direction = 'LC';
								nextdirection = 'RC';
							elseif i == 1
								direction = 'RC';
								nextdirection = 'LC';
							end
							
						else
							direction = nextdirection;
							nextdirection = [];
						end
						
					end
					
					% eightway
				elseif way == "eightway"
					if (azimuth >= -150) && (azimuth < -75)
						direction = "LM";
					elseif (azimuth >= -75) && (azimuth < -45)
						direction = "L";
					elseif (azimuth >= -45) && (azimuth < -30)
						direction = "LC";
					elseif (azimuth >= -30) && (azimuth < 0)
						direction = "CL";
					elseif (azimuth <= 150) && (azimuth > 75)
						direction = "RM";
					elseif (azimuth <= 75) && (azimuth > 45)
						direction = "R";
					elseif (azimuth <= 45) && (azimuth > 30)
						direction = "RC";
					elseif (azimuth <= 30) && (azimuth > 0)
						direction = "CR";
					else
						
						if isempty(nextdirection)
							i = rand(1);
							i = round(i);
							
							if i == 0
								direction = 'CL';
								nextdirection = 'CR';
							elseif i == 1
								direction = 'CR';
								nextdirection = 'CL';
							end
							
						else
							direction = nextdirection;
							nextdirection = [];
						end
						
					end
				end
				
				% fprintf("azimuth %02d direction %s\n", azimuth, direction)
				
				eeg = data.eeg{trial};
				leneeg = length(eeg) / data.fsample.eeg;
				%             lenwin = 5;
				if lenwin >= 5
					overlap = 8;
				else
					overlap = 4;
				end
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
						eegslice = eeg((slice - 1) * data.fsample.eeg * lenoverlap + (1:lenwin * data.fsample.eeg), :);
						SAVEPATH = fullfile(savepath, sprintf('CV_%02d', fold), sprintf('decision_win_%02.1fs', lenwin), 'train', direction);
						
						if ~isfolder(SAVEPATH)
							mkdir(SAVEPATH);
						end
						parsave(fullfile(SAVEPATH, sprintf('S%02d_trial_%02d_slice_%02d_way_%s', subject, trial, slice, direction)), eegslice);
					end
					
					for slice = validx
						eegslice = eeg((slice - 1) * data.fsample.eeg * lenoverlap + (1:lenwin * data.fsample.eeg), :);
						SAVEPATH = fullfile(savepath, sprintf('CV_%02d', fold), sprintf('decision_win_%02.1fs', lenwin), 'val', direction);
						
						if ~isfolder(SAVEPATH)
							mkdir(SAVEPATH);
						end
						
						parsave(fullfile(SAVEPATH, sprintf('S%02d_trial_%02d_slice_%02d_way_%s', subject, trial, slice, direction)), eegslice);
					end
					
					for slice = testidx
						eegslice = eeg((slice - 1) * data.fsample.eeg * lenoverlap + (1:lenwin * data.fsample.eeg), :);
						SAVEPATH = fullfile(savepath, sprintf('CV_%02d', fold), sprintf('decision_win_%02.1fs', lenwin), 'test', direction);
						
						if ~isfolder(SAVEPATH)
							mkdir(SAVEPATH);
						end
						
						parsave(fullfile(SAVEPATH, sprintf('S%02d_trial_%02d_slice_%02d_way_%s', subject, trial, slice, direction)), eegslice);
					end
					fprintf("Subject %02d trial %02d fold %02d complete\n", subject, trial, fold)
				end
				
			end
			
		end
	end
end

function precompute_edge_boxes(min_id, max_id)
    globals;
    suncgDir = suncgDir; % redundancy useful for parfor

    addpath(genpath('./matUtils'));
    basedir = pwd();
    fileNamesAll = strsplit(fileread(fullfile(suncgDir, 'zipfiles', 'data_goodlist_v2.txt')), '\n');

    addpath(genpath('./matUtils'));
    addpath(genpath('../../external/edges/'));
    addpath(genpath('../../external/toolbox/'));

    basedir = pwd();
    saveDir = fullfile(suncgDir, 'edgebox_proposals');
    nodeDir = fullfile(suncgDir, 'bboxes_node');
    mkdirOptional(saveDir);

    sceneIds = getFileNamesFromDirectory(fullfile(suncgDir, 'camera'),'types',{''});
    sceneIds = sceneIds(3:end);
    sceneIds = sort(sceneIds);
    if max_id == 0
        max_id = length(sceneIds);
    end

    %% load pre-trained edge detection model and set opts (see edgesDemo.m)
    model=load('../../external/edges/models/forest/modelBsds'); model=model.model;
    model.opts.multiscale=0; model.opts.sharpen=2; model.opts.nThreads=4;

    %% set up opts for edgeBoxes (see edgeBoxes.m)
    opts = edgeBoxes;
    opts.alpha = .65;     % step size of sliding window search
    opts.beta  = .75;     % nms threshold for object proposals
    opts.minScore = .01;  % min score of boxes to detect
    opts.maxBoxes = 1e3;  % max number of boxes to detect
    
    for ix = min_id:max_id
    %for ix = min_id:max_id
        if mod(ix, 100) == 0
            disp(ix)
        end
        sceneId = sceneIds{ix};
        mkdirOptional(fullfile(saveDir, sceneId));
        imgsAll = getFileNamesFromDirectory(fullfile(suncgDir, 'renderings_node', sceneId),'types',{'.png'});

        for cameraId=1:length(imgsAll)
            saveFile = fullfile(saveDir, sceneId, sprintf('%06d_proposals.mat', cameraId-1));
            if exist(saveFile, 'file')
                continue
            end
            if ~ismember(sprintf('%s/%06d', sceneId, cameraId-1), fileNamesAll)
                continue
            end
            img_file = fullfile(suncgDir, 'renderings_ldr', sceneId, sprintf('%06d_mlt.png', cameraId-1));
            nodeFile = fullfile(nodeDir, sceneId, sprintf('%06d_bboxes.mat', cameraId-1));

            if ~exist(img_file, 'file')
                % Bad file
                disp(img_file);
                continue
            end

            % disp(saveFile);
            img = imread(img_file);
            var = load(nodeFile);
            prop=edgeBoxes(img,model,opts);
            proposals = prop;
            proposals(:,3:4) = proposals(:,3:4) + proposals(:,1:2);
            
            overlaps = bboxOverlap(var.bboxes, proposals(:,1:4));
            overlapsGt = (max(overlaps, [], 2) > 0.7);
            [overlapsProposals, gtInds] = max(overlaps, [], 1);

            saveFunc(saveFile, proposals, overlapsProposals, gtInds);
        end
    end
end

function saveFunc(filename, proposals, overlaps, gtInds)
    save(filename,'proposals', 'overlaps', 'gtInds');
end
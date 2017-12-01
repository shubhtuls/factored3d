function precompute_gt_bboxes(min_id, max_id)
    globals;
    suncgDir = suncgDir; % redundancy useful for parfor
    addpath(genpath('./matUtils'));
    basedir = pwd();
    fileNamesAll = strsplit(fileread(fullfile(suncgDir, 'zipfiles', 'data_goodlist_v2.txt')), '\n');

    saveDir = fullfile(suncgDir, 'bboxes_node');
    mkdirOptional(saveDir);

    sceneIds = getFileNamesFromDirectory(fullfile(suncgDir, 'camera'),'types',{''});
    sceneIds = sceneIds(3:end);
    sceneIds = sort(sceneIds);
    if max_id == 0
        max_id = length(sceneIds);
    end
    parfor ix = min_id:max_id
    %for ix = min_id:max_id
        sceneId = sceneIds{ix};
        nodesBoxesDir = fullfile(saveDir, sceneId);
        mkdirOptional(nodesBoxesDir);
        imgsAll = getFileNamesFromDirectory(fullfile(suncgDir, 'renderings_node', sceneId),'types',{'.png'});

        for cameraId=1:length(imgsAll)
            if ~ismember(sprintf('%s/%06d', sceneId, cameraId-1), fileNamesAll)
                continue
            end
            img = imread(fullfile(suncgDir, 'renderings_node', sceneId, sprintf('%06d_node.png', cameraId-1)));
            ids = unique(img);
            nIds = size(ids,1);
            bboxes = zeros(nIds,4);
            nPixels = zeros(nIds,1);
            for o=1:nIds
                bboxes(o,:) = mask2bbox(img == ids(o));
                nPixels(o,:) = sum(sum(img == ids(o)));
            end
            saveFile = fullfile(nodesBoxesDir, sprintf('%06d_bboxes.mat', cameraId-1));
            saveFunc(saveFile, ids, bboxes, nPixels);
        end
    end
end

function saveFunc(filename, ids, bboxes, nPixels)
    save(filename,'ids', 'bboxes', 'nPixels');
end

function bbox = mask2bbox(mask)
    [y,x] = find(mask);
    bbox = [min(x) min(y) max(x) max(y)];
end
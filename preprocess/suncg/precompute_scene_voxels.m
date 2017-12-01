function precompute_scene_voxels(min_id, max_id)
    globals;
    suncgDir = suncgDir; % redundancy useful for parfor
    addpath(genpath('./matUtils'));
    basedir = pwd();
    fileNamesAll = strsplit(fileread(fullfile(suncgDir, 'zipfiles', 'data_goodlist_v2.txt')), '\n');

    sscnetDir = fullfile(basedir, '..', '..', 'external', 'sscnet', 'matlab_code');
    addpath(sscnetDir);

    objectcategory = load(fullfile(sscnetDir, 'suncgObjcategory.mat'));
    addpath(fullfile(sscnetDir, 'utils'));
    addpath(fullfile(basedir, 'matUtils'));
    sceneIds = getFileNamesFromDirectory(fullfile(suncgDir, 'camera'),'types',{''});
    sceneIds = sceneIds(3:end);
    sceneIds = sort(sceneIds);

    if max_id == 0
        max_id = length(sceneIds);
    end
    parfor ix = min_id:max_id
        genSceneData(sceneIds{ix}, suncgDir, objectcategory.objcategory, fileNamesAll);
    end
end


function genSceneData(sceneId, suncgDir, objcategory, fileNamesAll)
    %% generating scene voxels in camera view
    camerafile = sprintf('%s/camera/%s/room_camera.txt', suncgDir, sceneId);
    cameraInfofile = sprintf('%s/camera/%s/room_camera_name.txt', suncgDir, sceneId);
    cameraInfo = readCameraName(cameraInfofile);
    cameraPoses = readCameraPose(camerafile);
    voxPath = fullfile(suncgDir, 'scene_voxels', sceneId);
    mkdirOptional(voxPath);

    for cameraId = 1:length(cameraInfo)
        if ~ismember(sprintf('%s/%06d', sceneId, cameraId-1), fileNamesAll)
            continue
        end
        sceneVoxMatFilename = fullfile(voxPath,sprintf('%06d_voxels.mat',cameraId-1));
        sceneVoxFilename = [sceneVoxMatFilename(1:(end-4)),'.bin'];
        if exist(sceneVoxMatFilename, 'file')
            continue
        end

        % get camera extrisic yup -> zup
        extCam2World = camPose2Extrinsics(cameraPoses(cameraId,:));
        extCam2World = [[1 0 0; 0 0 1; 0 1 0]*extCam2World(1:3,1:3) extCam2World([1,3,2],4)];

        % generating scene voxels in camera view 
        [sceneVox] = get_scene_vox(suncgDir,sceneId,cameraInfo(cameraId).floorId+1,cameraInfo(cameraId).roomId+1,extCam2World,objcategory);
        camPoseArr = [extCam2World',[0;0;0;1]]; %'
        % camPoseArr = camPoseArr(:);
        sceneVox = (sceneVox ~= 0) & (sceneVox ~= 255);

        % Compress with RLE and save to binary file 
        % writeRLEfile(sceneVoxFilename, sceneVox,camPoseArr,voxOriginWorld)
        save(sceneVoxMatFilename,'sceneVox','camPoseArr');
    end
end
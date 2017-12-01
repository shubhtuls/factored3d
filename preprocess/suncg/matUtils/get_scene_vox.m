function [sceneVox] = get_scene_vox(pathToData,sceneId,floorId,roomId,extCam2World,objcategory)
% Notes: grid is Z up while the The loaded houses are Y up
% Adapted from the sscnet codebase - https://github.com/shurans/sscnet

volume_params;
ignore_classes = {'people', 'plants'};
% Compute voxel range in cam coordinates
voxOriginCam = - [voxSize(1)/2*voxUnit;voxSize(2)/2*voxUnit;0];
[gridPtsCamX,gridPtsCamY,gridPtsCamZ] = ndgrid(voxOriginCam(1):voxUnit:(voxOriginCam(1)+(voxSize(1)-1)*voxUnit), ...
                                               voxOriginCam(2):voxUnit:(voxOriginCam(2)+(voxSize(2)-1)*voxUnit), ...
                                               voxOriginCam(3):voxUnit:(voxOriginCam(3)+(voxSize(3)-1)*voxUnit));
gridPtsCam_init = [gridPtsCamX(:),gridPtsCamY(:),gridPtsCamZ(:)]'; %'

% Compute voxel grid centres in world coordinates
gridPtsWorld = bsxfun(@plus,extCam2World(1:3,1:3)*gridPtsCam_init, extCam2World(1:3,4));
gridPtsWorldX = gridPtsWorld(1,:);
gridPtsWorldY = gridPtsWorld(2,:);
gridPtsWorldZ = gridPtsWorld(3,:);
gridPtsLabel = zeros(1,size(gridPtsWorld,2));

house = loadjson(fullfile(pathToData,'house', sceneId,'house.json'));
roomStruct = house.levels{floorId}.nodes{roomId};
floorStruct = house.levels{floorId};

% find all grid in the room 
floorObj = read_wobj_safe([fullfile(pathToData,'room',sceneId,roomStruct.modelId) 'f.obj']);
inRoom = zeros(size(gridPtsWorldX));
for i = 1:length(floorObj.objects(3).data.vertices)
    faceId = floorObj.objects(3).data.vertices(i,:);
    floorP = floorObj.vertices(faceId,[1,3])';
    inRoom = inRoom|inpolygon(gridPtsWorldX,gridPtsWorldY,floorP(1,:),floorP(2,:)); %'
end

% find floor 
floorZ = mean(floorObj.vertices(:,2));
gridPtsObjWorldInd = inRoom(:)'&(abs(gridPtsWorld(3,:)-floorZ) <= voxUnit/2); %'
[~,classRootId] = getobjclassSUNCG('floor',objcategory);
gridPtsLabel(gridPtsObjWorldInd) = classRootId;  

% find ceiling 
ceilObj = read_wobj_safe([fullfile(pathToData,'room',sceneId,roomStruct.modelId) 'c.obj']);
ceilZ = mean(ceilObj.vertices(:,2));
gridPtsObjWorldInd = inRoom(:)'&abs(gridPtsWorld(3,:)-ceilZ) <= voxUnit/2; %'
[~,classRootId] = getobjclassSUNCG('ceiling',objcategory);
gridPtsLabel(gridPtsObjWorldInd) = classRootId;  

% Load walls
WallObj = read_wobj_safe([fullfile(pathToData,'room',sceneId,roomStruct.modelId) 'w.obj']);
inWall = zeros(size(gridPtsWorldX));
for oi = 1:length(WallObj.objects)
    if WallObj.objects(oi).type == 'f'
        for i = 1:length(WallObj.objects(oi).data.vertices)
            faceId = WallObj.objects(oi).data.vertices(i,:);
            floorP = WallObj.vertices(faceId,[1,3])'; %'
            inWall = inWall|inpolygon(gridPtsWorldX,gridPtsWorldY,floorP(1,:),floorP(2,:));
        end
    end
end
gridPtsObjWorldInd = inWall(:)'&(gridPtsWorld(3,:)<ceilZ-voxUnit/2)&(gridPtsWorld(3,:)>floorZ+voxUnit/2); %'
[~,classRootId] = getobjclassSUNCG('wall',objcategory);
gridPtsLabel(gridPtsObjWorldInd) = classRootId;     

% Loop through each object and set voxels to class ID
for objId = roomStruct.nodeIndices
    object_struct = floorStruct.nodes{objId+1};
    if isfield(object_struct, 'modelId') && isfield(object_struct, 'valid') && (object_struct.valid)
        % Set segmentation class ID
        [classRootName,classRootId,className] = getobjclassSUNCG(strrep(object_struct.modelId,'/','__'),objcategory);
        if ismember(className, ignore_classes)
            continue
        end

        % Compute object bbox in world coordinates
        objBbox = [object_struct.bbox.min([1,3,2])',object_struct.bbox.max([1,3,2])'];

        % Load segmentation of object in object coordinates
        filename= fullfile(pathToData,'object_vox/object_vox_data/',strrep(object_struct.modelId,'/','__'), [strrep(object_struct.modelId,'/','__'), '.binvox']);
        [voxels,scale,translate] = read_binvox(filename);
        [x,y,z] = ind2sub(size(voxels),find(voxels(:)>0));   
        objSegPts = bsxfun(@plus,[x,y,z]*scale,translate'); %'

        % Convert object to world coordinates
        extObj2World_yup = reshape(object_struct.transform,[4,4]);
        objSegPts = extObj2World_yup*[objSegPts(:,[1,3,2])';ones(1,size(x,1))]; %'
        objSegPts = objSegPts([1,3,2],:);

        % Get all grid points within the object bbox in world coordinates
        gridPtsObjWorldInd =      gridPtsWorld(1,:) >= objBbox(1,1) - voxUnit & gridPtsWorld(1,:) <= objBbox(1,2) + voxUnit & ...
                                  gridPtsWorld(2,:) >= objBbox(2,1) - voxUnit & gridPtsWorld(2,:) <= objBbox(2,2) + voxUnit & ...
                                  gridPtsWorld(3,:) >= objBbox(3,1) - voxUnit & gridPtsWorld(3,:) <= objBbox(3,2) + voxUnit;
        gridPtsObjWorld = gridPtsWorld(:,find(gridPtsObjWorldInd));


        % If object is a window or door, clear voxels in object bbox
        [~,wallId] = getobjclassSUNCG('wall',objcategory); 
        if classRootId == 4 || classRootId == 5
           gridPtsObjClearInd = gridPtsObjWorldInd&gridPtsLabel==wallId;
           gridPtsLabel(gridPtsObjClearInd) = 0;
        end

        % Apply segmentation to grid points of object
        if numel(gridPtsObjWorld) > 0
            [indices, dists] = multiQueryKNNSearchImpl(pointCloud(objSegPts'), gridPtsObjWorld',1);
            objOccInd = find(sqrt(dists) <= (sqrt(3)/2)*scale);
            gridPtsObjWorldLinearIdx = find(gridPtsObjWorldInd);
            gridPtsLabel(gridPtsObjWorldLinearIdx(objOccInd)) = classRootId;
        end
    end
end

% Remove grid points not in field of view
extWorld2Cam = inv([extCam2World;[0,0,0,1]]);
gridPtsCam = extWorld2Cam(1:3,1:3)*gridPtsWorld + repmat(extWorld2Cam(1:3,4),1,size(gridPtsWorld,2));
gridPtsPixX = gridPtsCam(1,:).*(camK(1,1))./gridPtsCam(3,:)+camK(1,3);
gridPtsPixY = gridPtsCam(2,:).*(camK(2,2))./gridPtsCam(3,:)+camK(2,3);
invalidPixInd = (gridPtsPixX < 0 | gridPtsPixX >= im_w | gridPtsPixY < 0 | gridPtsPixY >= im_h | gridPtsCam(3,:) < 0);
gridPtsLabel(find(invalidPixInd)) = 0;

% Remove grid points not in the room
gridPtsLabel(~inRoom(:)&gridPtsLabel(:)==0) = 255;

% Save the volume
sceneVox = reshape(gridPtsLabel,voxSize'); %'

end
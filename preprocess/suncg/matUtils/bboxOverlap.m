function [iu i a1 a2] = bboxOverlap(B1, B2)
% function [iu i a1 a2] = bboxOverlap(B1, B2)
%   B1 and B2 are N x 4 and M x 4 matrices with values [xmin ymin xmax ymax] quadtruples
%   iu is N x M matrix with intersection over union, i is N x M matrix of the intersection
%   a1 is the area of boxes in B1 and a2 is the area of boxes in B2.

% AUTORIGHTS
% ---------------------------------------------------------
% Copyright (c) 2014, Saurabh Gupta
% 
% This file is part of the Utils code and is available 
% under the terms of the Simplified BSD License provided in 
% LICENSE. Please retain this notice and LICENSE if you use 
% this file (or any portion of it) in your project.
% ---------------------------------------------------------
  if(numel(B1) == 0 && numel(B2) == 0)
    a1 = zeros(0,0); a2 = zeros(0,0); i = zeros(0,0); iu = zeros(0,0);
  elseif(numel(B1) == 0)
    a1 = zeros(0,0); i = zeros(0,size(B2,1)); iu = zeros(0,size(B2,1));
    a2 = (B2(:,3)-B2(:,1)+1).*(B2(:,4)-B2(:,2)+1);

  elseif(numel(B2) == 0)
    a2 = zeros(0,0); i = zeros(size(B1,1), 0); iu = zeros(size(B1,1), 0);
    a1 = (B1(:,3)-B1(:,1)+1).*(B1(:,4)-B1(:,2)+1);

  else
    a1 = (B1(:,3)-B1(:,1)+1).*(B1(:,4)-B1(:,2)+1);
    a2 = (B2(:,3)-B2(:,1)+1).*(B2(:,4)-B2(:,2)+1);
    
    minX = bsxfun(@max, B1(:,1), B2(:,1)');
    minY = bsxfun(@max, B1(:,2), B2(:,2)');

    maxX = bsxfun(@min, B1(:,3), B2(:,3)');
    maxY = bsxfun(@min, B1(:,4), B2(:,4)');
    
    i = max(maxX-minX+1, 0).*max(maxY-minY+1, 0);
    iu = i./max(eps, bsxfun(@plus, a1, a2')-i);
  end
end

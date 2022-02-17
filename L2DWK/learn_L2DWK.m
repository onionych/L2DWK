function Weight = learn_L2DWK(R,target,kt,rc,div,lambda)
% learn L2DWK  
% parameter:
% Weight: Weights of classifiers 
% R     : the classification result, where the i-th data is classified R_{ij} by the j-th classifier   
% target: an N*1 vector, N is the number of data
% kt    : kernel type, include{'linear','guass','poly'}
% rc    : parameter for selected kernel type
% div   : diversity type, inclue{'dis','df'};
% lambda: parameter for loss - diversity combinition

if nargin <5
div = 'dis';
end

if nargin <6
lambda =1;
end

switch(div)
    case {'dis'}
        Weight = L2DWK_Dis(R,target,kt,rc,lambda);
    case {'df'}
         Weight = L2DWK_Df(R,target,kt,rc,lambda);
    otherwise 
        L = size(R,2);
        Weight = 1/L*ones(L,1);
end
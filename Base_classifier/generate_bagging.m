function [classifiers, R] = generate_bagging(traininput, traintarget, L, base_type)
% Generate bagging classifiers.
% parameter:
% classifiers: output the training bagging classifiers
% R          : the classification result, where the i-th data is classified O_{ij} by the j-th classifier   
% traininput : an N*Ni matrix, where N is the number of data, and Ni is the number of feature
% traintarget: an N*1 vector.
% L 	   	 : number of classifiers
% base_type  : the type of base classifier, include {'tree','nerual network', 'naive bayes'}, default 'tree'

if nargin < 3
	L=31;
end

if nargin < 4
	base_type ='tree';
end

classifiers = cell(1,L);
TN = length(traintarget);
R = zeros(TN,L);

n =0 ;
for p=1:L;
    fprintf(1,repmat('\b',1,n));
 	n = fprintf(1,'%d',p);
    rp = (floor(TN*rand(1,TN))+1);
	classifiers{p} = train_base_classifier(traininput(:,rp)',traintarget(rp)',base_type);
	R(:,p) = predict_base_classifier(classifiers{p},traininput',base_type);
end
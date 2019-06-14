function predict = predict_bagging(classifiers, testinput, base_type)
% predict the Bagging result for each classifier
% parameter:
% predict    : the classification result of testing data, where the i-th data is classified predict_{ij} by the j-th classifier 
% classifiers: bagging classifiers
% testinput  : an N*Ni matrix, where N is the number of data, and Ni is the number of feature
% base_type  : the type of base classifier, include {'tree','nerual network', 'naive bayes'}, default 'tree'

if nargin < 3
	base_type ='tree';
end

predict = zeros(size(testinput,2),length(classifiers));

for p=1:length(classifiers);
 	predict(:,p) = predict_base_classifier(classifiers{p},testinput',base_type);
end
function predict_ = predict_base_classifier(base_classifier, testinput, base_type)
% predict by base classifier 
%
% parameter:
% predict 			: predict result, an N*1 vector. 
% base_classifier	: base classifier
% testinput 		: an N*Ni matrix, where N is the number of data, and Ni is the number of feature
% base_type  		: the type of base classifier, include {'tree','nerual network', 'naive bayes'}, default 'tree'

if nargin < 3
    base_type = 'tree';
end

switch(base_type)
    case {'tree'}
        predict_ = predict(base_classifier,testinput);
    
	case {'nerual_network'}
	    y = sim(base_classifier,testinput');
        [~,predict_]=max(y);
		predict_ = predict_';
        
    case{'naive_bayes'}
        predict_ = base_classifier.predict(testinput);
    otherwise
        predict_ = zeros(size(testinput,2),1);
end

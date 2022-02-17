function base_classifier = train_base_classifier(traininput, traintarget, base_type)
% Generate base classifier
%
% parameter:
% base_classifier	: output the training base classifier
% traininput 		: an N*Ni matrix, where N is the number of data, and Ni is the number of feature
% traintarget		: an N*1 vector.
% base_type  		: the type of base classifier, include {'tree','nerual network', 'naive bayes'}, default 'tree'


if nargin < 3
    base_type = 'tree';
end

switch(base_type)
    case {'tree'}
        base_classifier = fitctree(traininput,traintarget);
    
	case {'nerual_network'}
	    C = unique(traintarget);
		if(size(C,1) > 1)
			C=C';
		end
		classno = length(C);
        N = size(traininput,1);
		
		% calculate the number of hidden node
		sl = floor(N/10/(size(traininput,1)+length(C)))+1;
		   
        T = double(repmat(traintarget',length(C),1)==repmat(C',1,N));
           
        base_classifier = newff(traininput',T,sl);
        base_classifier.trainParam.epochs=100;
        base_classifier.trainParam.showWindow = false;
        base_classifier.trainParam.show =50;
        base_classifier = train(base_classifier,traininput',T);
		
    case{'naive_bayes'}
        base_classifier = fitcnb(traininput,traintarget,'DistributionNames','mvmn');
    otherwise
        base_classifier = [];
end

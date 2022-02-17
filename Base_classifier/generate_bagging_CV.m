function generate_bagging_CV(dataname, L, CV, base_type, folderpath)
% Generate classification neural network base classifier,  CV folders,  L classifiers for each folders. 
% parameter:
% dataname   : load "dataname".mat
% L 	     : number of classifiers
% CV         : number of cross validation folders,default 10
% base_type  : the type of base classifier, include {'tree','nerual network', 'naive bayes'}, default 'tree'
% folderpath : path prefix for saving results 
%
% output: 3 Mat file
% _model.mat: store the classifiers and base_type
%    _Re.mat: store R, predict, traintarget, testtarget
%      -R   : the classification result of training data, where the i-th data is classified R_{ij} by the j-th classifier   
%   -predict: the classification result of testing data, where the i-th data is classified predict_{ij} by the j-th classifier   
%   _rec.mat: store rec_c, rec_e
%     -rec_c: recognition rate for each classifier
%     -rec_e: recognition rate for first k classifiers ensembles

if nargin < 2
L =101;
end 

if nargin < 3
CV = 10;
end

if nargin < 4
	base_type ='tree';
end

if nargin < 5
folderpath = dataname;
else 
folderpath = strcat(folderpath,'\',dataname);
end

% make the folder for dataset 
load(dataname);
system(['mkdir ',folderpath]);

cd(folderpath)
cd

% delete unlabeled data
input = input(target>0,:)';
target = target(target >0)';

% cross validation
N = length(target);
if CV > 2 
I = floor(CV*rand(1,N))+1;
save CV I CV
else  % when CV=1, 60% for training, 40% for testing
I = rand(1,N);
save CV I CV
end
        
for f=1:CV;
	dirName = strcat('valid',dec2base(f,10));
	system(['mkdir ',dirName]);
        
    cd(dirName);
    cd
        
	% generate trainset, testset
	if(CV > 2)
	TI = find(I~=f);
	SI = find(I==f);
    else 
	TI = find(I>=.4);
	SI = find(I<.4);
	end
	save(dirName,'TI','SI');
		
    traininput = input(:,TI);
    traintarget = target(TI);
    testinput = input(:,SI);
    testtarget = target(SI);
                
    base_classifier = cell(1,L);
    TN = length(TI);
    SN = length(SI);
    C = unique(target);
    if(size(C,1) > 1)
        C=C';
    end
    classno = length(C);

    fprintf('Start Training...\n');
        
    OUT = zeros(SN,length(C));
    rec_c = zeros(1,L);			% rec_c: recognition rate for each classifier
    rec_e = zeros(1,L);			% rec_e: recognition rate for first k classifiers ensembles
    R = zeros(TN,L);
             
    [classifiers,R] = generate_bagging(traininput, traintarget, L, base_type);
    predict = predict_bagging(classifiers, testinput, base_type);

	OUT =0;
	for p=1:length(classifiers);
		rec_c(p) = sum(predict(:,p) == testtarget')/SN;
		
		OUT = OUT + (repmat(predict(:,p),1,length(C))==repmat(C,SN,1));
		[men,pos]=max(OUT,[],2);
		rec_e(p) = sum(pos == testtarget')/SN;   
	end
	
	save(strcat('classifier_',base_type,'_',dec2base(L,10),'_model'),'classifiers','base_type');
	save(strcat('classifier_',base_type,'_',dec2base(L,10),'_Re'),'R','predict','traintarget','testtarget');
	save(strcat('classifier_',base_type,'_',dec2base(L,10),'_rec'),'rec_e','rec_c');
	cd ..
    end
    cd ..
end

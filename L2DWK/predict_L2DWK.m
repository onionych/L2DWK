function predict = predict_L2DWK(bag_predict,Weight,C)
% predict the L2DWK result
% parameter:
% predict    : an N*1 vector, where N is the number of data
% bag_predict: the classification result of testing data, where the i-th data is classified predict_{ij} by the j-th classifier 
% Weight 	 : classifier weights
% C			 : number of data class

OUT =0;
for p=1:size(bag_predict,2);
    OUT = OUT + Weight(p) * double(repmat(bag_predict(:,p),1,length(C))==repmat(C,size(bag_predict,1),1));
end

[~,predict] = max(OUT,[],2);
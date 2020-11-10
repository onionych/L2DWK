function Weight = L2DWK_Df(R,target,kt,rc,lambda)
% L2DWK with double fault diversity 
% parameter:
% Weight: Weights of classifiers 
% R     : the classification result, where the i-th data is classified R_{ij} by the j-th classifier    
% target: an N*1 vector, N is the number of data
% kt    : kernel type, include{'linear','guass','poly'}
% rc    : parameter for selected kernel type
% lambda: parameter for loss - diversity combinition

if nargin <5
lambda =1;
end

turn = 30;
e = ones(turn,1);
e_min = 1;

[N L] = size(R);
O  =  2*(R == repmat(target',1,L))-1;
C = unique(target);

if(size(C,1)>1)
    C=C';
end
        
Aeq = [0 ones(1,L)];
Beq = 1;
lb =  [1; zeros(L,1)];
ub =  [1; ones(L,1)];
V =  ones(N,1)/N;				%	sample weights
W0 = [1;ones(L,1)/L];
% W0(2:end) = rand(L,1);
% W0(2:end) = W0(2:end)/sum(W0(2:end)); %ones(L,1)/L];
op = ones(N,1);
warning off
opts = optimset('Display','off');

n =0;
for i=1:turn;
    %	Df
	f = km_kernel_U(O',op',kt,rc,V);
	d = km_kernel_U(1-O',1-O',kt,rc,V);
    
    H = [1 -lambda*f';-lambda*f -d];
    
    [~,u] = chol(H);
    while(u~= 0 )
        [HV,HD] =eig(H);
        J = diag(HD);
        J(J<=0) = 1e-6;
        HD=diag(J);
        H = real(HV*HD*HV');
        [~,u] = chol(H);
    end
    
    [W fval] = quadprog(H,zeros(size(H,1),1),[],[],Aeq,Beq,lb,ub,[],opts);

    if(isempty(W))
        W = W0;
    end
    
    OUT_W = 0;
    for p=2:L+1;
        OUT_W = OUT_W + W(p)*(repmat(R(:,p-1),1,length(C))==repmat(C,N,1));
    end
    [men,pos] = max(OUT_W,[],2);
    
    s = (O*W(2:end) <=0);
    p = double(s);
    
    e(i) = mean(double(pos ~= target'));
    
    if(e(i)==0)
        W0 = W ;
        break;
    end
    
    V =1/i* p/e(i) + V;
%     alpha = log((1-e(i))/e(i))+log(length(C)-1);
%     V = V.*exp(alpha*(p));
    V = V/sum(V); 
    
    fprintf(1,repmat('\b',1,n));
 	n = fprintf(1,'%d',i);
    
    if(e(i) <= e_min)
        e_min = e(i);
        W0 = W ;
    else
        continue;
    end
end

Weight = W0(2:end).*(W0(2:end)>.001);

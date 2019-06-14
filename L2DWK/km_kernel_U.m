function K = km_kernel_U(X1,X2,ktype,kpar,U)
% KM_KERNEL calculates the kernel matrix with data sample.
% Input:	- X1, X2: data matrices in row format (data as rows)
%			- ktype: string representing kernel type
%			- kpar: vector containing the kernel parameters
%           - U: data sample
% Output:	- K: kernel matrix
% USAGE: K = km_kernel(X1,X2,ktype,kpar,U)
%
% Author: Steven Van Vaerenbergh (steven *at* gtas.dicom.unican.es), 2012.
% Id: km_kernel.m v1.2 20120513
% This file is part of the Kernel Methods Toolbox (KMBOX) for MATLAB.
% http://sourceforge.net/p/kmbox
%
% This program is free software: you can redistribute it and/or modify it
% under the terms of the GNU General Public License as published by the
% Free Software Foundation, version 3 (http://www.gnu.org/licenses).

if nargin < 5
    U = ones(size(X1,2),1);
end

switch ktype
	case 'gauss'	% Gaussian kernel
		sgm = kpar;	% kernel width
		
		dim1 = size(X1,1);
		dim2 = size(X2,1);
		
		norms1 = X1.^2*U;
		norms2 = X2.^2*U;
		
		mat1 = repmat(norms1,1,dim2);
		mat2 = repmat(norms2',dim1,1);
		
		distmat = mat1 + mat2 - 2*X1*diag(U)*X2';	% full distance matrix
		K = exp(-distmat/(2*sgm^2));
		
	%case 'gauss-diag'	% only diagonal of Gaussian kernel
%		sgm = kpar;	% kernel width
%		K = exp(-sum((X1-X2).^2,2)/(2*sgm^2));
		
	case 'poly'	% polynomial kernel
		p = kpar(1);	% polynome order
		c = kpar(2);	% additive constant
		
		K = (X1*diag(U)*X2' + c).^p;
		
	case 'linear' % linear kernel
		K = X1*diag(U)*X2';
		
	otherwise	% default case
		error ('unknown kernel type')
end
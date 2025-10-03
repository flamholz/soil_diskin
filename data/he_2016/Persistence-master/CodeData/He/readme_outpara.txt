        load(strcat('./',mdl{n},'/',mdl{n},'_FixClim1_for3boxmodel.mat'));
        mdlstatpath = strcat('./',mdl{n},'/Extrapolate_D14CSOC_3pool/3poolmodel_3pooldata/outpara');
        para = importdata(mdlstatpath);

        [row,col,yr] = size(annufVegSoil);
        [ii,jj] = ind2sub([row,col],para(:,1));

        tau_fast = zeros(row,col);
        tau_medium = zeros(row,col);
        tau_passive = zeros(row,col);        
        rf = zeros(row,col);
        rs = zeros(row,col);
        for k = 1:size(ii,1)
            tau_fast(ii(k),jj(k)) = para(k,2);
            tau_medium(ii(k),jj(k)) = para(k,3);
            tau_passive(ii(k),jj(k)) = para(k,4);
            rf(ii(k),jj(k)) = para(k,5);
            rs(ii(k),jj(k)) = para(k,6);
        end
        tau_fast(tau_fast==0) = nan;
        tau_medium(tau_medium==0) = nan;
        tau_passive(tau_passive==0) = nan;        
        rf(rf==0) = nan;
        rs(rs==0) = nan;
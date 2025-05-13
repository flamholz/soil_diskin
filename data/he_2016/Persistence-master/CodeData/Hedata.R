
library(R.matlab)
library(SoilR)

dat=readMat("~/SOIL-R/Manuscripts/Persistence/CodeData/He/CESM/CESM_FixClim1_for3boxmodel.mat")
fl=dat$annufVegSoil
mean_fl=apply(fl, c(1,2), mean, na.rm=TRUE) # Mean across 140 years

flv=as.numeric(mean_fl)
flid=na.omit(flv)*86400*365 # Convert from kg C m-2 sec-1 to kg C m-2 yr-1
ids=which(!is.na(flv))

inputs=data.frame(ids,flid)

pars=read.table("~/SOIL-R/Manuscripts/Persistence/CodeData/He/CESM/outpara")

tmp=pars[order(pars[,1]),] # Ordered parameter values per id

matchids=match(tmp[,1],inputs[,1])

CESM=data.frame(tmp,inputs[matchids,2])
names(CESM)<-c("id", "tau1", "tau2", "tau3", "rf", "rs", "u")

write.table(CESM,"~/SOIL-R/Manuscripts/Persistence/CodeData/He/CESM/compartmentalParameters.txt", row.names = FALSE )

A1=diag(-1/CESM[1111,2:4])
A1[2,1]=CESM[1111,5]/CESM[1111,2]
A1[3,2]=CESM[1111,6]/CESM[1111,3]

tau=seq(0,1000)
age1=systemAge(A=A1, u=matrix(c(CESM[1111,7],0,0),3,1), a=tau)
trt1=transitTime(A=A1, u=matrix(c(CESM[1111,7],0,0),3,1), a=tau)

########
# IPSL
dat=readMat("~/SOIL-R/Manuscripts/Persistence/CodeData/He/IPSL/IPSL_FixClim1_for3boxmodel.mat")
fl=dat$annufVegSoil
mean_fl=apply(fl, c(1,2), mean, na.rm=TRUE) # Mean across 140 years

flv=as.numeric(mean_fl)
flid=na.omit(flv)*86400*365 # Convert from kg C m-2 sec-1 to kg C m-2 yr-1
ids=which(!is.na(flv))

inputs=data.frame(ids,flid)

pars=read.table("~/SOIL-R/Manuscripts/Persistence/CodeData/He/IPSL/outpara")

tmp=pars[order(pars[,1]),] # Ordered parameter values per id

matchids=match(tmp[,1],inputs[,1])

IPSL=data.frame(tmp,inputs[matchids,2])
names(IPSL)<-c("id", "tau1", "tau2", "tau3", "rf", "rs", "u")

write.table(IPSL,"~/SOIL-R/Manuscripts/Persistence/CodeData/He/IPSL/compartmentalParameters.txt", row.names = FALSE )

########
# MRI
dat=readMat("~/SOIL-R/Manuscripts/Persistence/CodeData/He/MRI/MRI_FixClim1_for3boxmodel.mat")
fl=dat$annufVegSoil
mean_fl=apply(fl, c(1,2), mean, na.rm=TRUE) # Mean across 140 years

flv=as.numeric(mean_fl)
flvz=flv[flv != 0]
flid=flvz*86400*365 # Convert from kg C m-2 sec-1 to kg C m-2 yr-1
ids=which(flv != 0)

inputs=data.frame(ids,flid)

pars=read.table("~/SOIL-R/Manuscripts/Persistence/CodeData/He/MRI/outpara")

tmp=pars[order(pars[,1]),] # Ordered parameter values per id

matchids=match(tmp[,1],inputs[,1])

MRI=data.frame(tmp,inputs[matchids,2])
names(MRI)<-c("id", "tau1", "tau2", "tau3", "rf", "rs", "u")

write.table(MRI,"~/SOIL-R/Manuscripts/Persistence/CodeData/He/MRI/compartmentalParameters.txt", row.names = FALSE )


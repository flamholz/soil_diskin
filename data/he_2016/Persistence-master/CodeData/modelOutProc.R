library(R.matlab)
library(raster)
library(rasterVis)

# CESM
cesmpar=read.table("~/SOIL-R/Manuscripts/Persistence/CodeData/He/CESM/compartmentalParameters.txt", header = TRUE)
cesmout=read.csv("~/SOIL-R/Manuscripts/Persistence/CodeData/He/CESM/CESMages.csv")

datcesm=readMat("~/SOIL-R/Manuscripts/Persistence/CodeData/He/CESM/CESM_FixClim1_for3boxmodel.mat")
flcesm=datcesm$annufVegSoil
rm(datcesm) # Don't need to store this object in memory

cesmTT=matrix(NA, nrow = dim(flcesm[,,1])[1], ncol=dim(flcesm[,,1])[2])
cesmTT[cesmpar$id]=cesmout$meanTransitTime
cesmTTr=raster(cesmTT)

cesmA=matrix(NA, nrow = dim(flcesm[,,1])[1], ncol=dim(flcesm[,,1])[2])
cesmA[cesmpar$id]=cesmout$meanSystemAge
cesmAr=raster(cesmA)

cesmP95=matrix(NA, nrow = dim(flcesm[,,1])[1], ncol=dim(flcesm[,,1])[2])
cesmP95[cesmpar$id]=cesmout$P95
cesmP95r=raster(cesmP95)

cesmbrick=brick(list(cesmTTr, cesmAr))

pdf("~/SOIL-R/Manuscripts/Persistence/Figures/sattCESM.pdf")
levelplot(cesmbrick, names.attr=c("Mean trainsit time", "Mean age"), main="CESM")
dev.off()

pdf("~/SOIL-R/Manuscripts/Persistence/Figures/ttCESM.pdf")
levelplot(cesmTTr, main="Mean transit time: CESM")
dev.off()

pdf("~/SOIL-R/Manuscripts/Persistence/Figures/saCESM.pdf")
levelplot(cesmAr, main="Mean SOM age: CESM")
dev.off()

pdf("~/SOIL-R/Manuscripts/Persistence/Figures/P95CESM.pdf")
levelplot(cesmP95r, main="CESM")
dev.off()

# IPSL
ipslpar=read.table("~/SOIL-R/Manuscripts/Persistence/CodeData/He/IPSL/compartmentalParameters.txt", header = TRUE)
ipslout=read.csv("~/SOIL-R/Manuscripts/Persistence/CodeData/He/IPSL/IPSLages.csv")

datipsl=readMat("~/SOIL-R/Manuscripts/Persistence/CodeData/He/IPSL/IPSL_FixClim1_for3boxmodel.mat")
flipsl=datipsl$annufVegSoil
rm(datipsl)

ipslTT=matrix(NA, nrow = dim(flipsl[,,1])[1], ncol=dim(flipsl[,,1])[2])
ipslTT[ipslpar$id]=ipslout$meanTransitTime
ipslTTr=raster(ipslTT)

ipslA=matrix(NA, nrow = dim(flipsl[,,1])[1], ncol=dim(flipsl[,,1])[2])
ipslA[ipslpar$id]=ipslout$meanSystemAge
ipslAr=raster(ipslA)

ipslP95=matrix(NA, nrow = dim(flipsl[,,1])[1], ncol=dim(flipsl[,,1])[2])
ipslP95[ipslpar$id]=ipslout$P95
ipslP95r=raster(ipslP95)

ipslbrick=brick(list(ipslTTr, ipslAr))

pdf("~/SOIL-R/Manuscripts/Persistence/Figures/sattIPSL.pdf")
levelplot(ipslbrick, names.attr=c("Mean trainsit time", "Mean age"), main="IPSL")
dev.off()

pdf("~/SOIL-R/Manuscripts/Persistence/P95IPSL.pdf")
levelplot(ipslP95r, main="IPSL")
dev.off()

# MRI
mripar=read.table("~/SOIL-R/Manuscripts/Persistence/CodeData/He/MRI/compartmentalParameters.txt", header = TRUE)
mriout=read.csv("~/SOIL-R/Manuscripts/Persistence/CodeData/He/MRI/MRIages.csv")

datmri=readMat("~/SOIL-R/Manuscripts/Persistence/CodeData/He/MRI/MRI_FixClim1_for3boxmodel.mat")
flmri=datmri$annufVegSoil
rm(datmri)

mriTT=matrix(NA, nrow = dim(flmri[,,1])[1], ncol=dim(flmri[,,1])[2])
mriTT[mripar$id]=mriout$meanTransitTime
mriTTr=raster(mriTT)

mriA=matrix(NA, nrow = dim(flmri[,,1])[1], ncol=dim(flmri[,,1])[2])
mriA[mripar$id]=mriout$meanSystemAge
mriAr=raster(mriA)

mriP95=matrix(NA, nrow = dim(flmri[,,1])[1], ncol=dim(flmri[,,1])[2])
mriP95[mripar$id]=mriout$P95
mriP95r=raster(mriP95)

mribrick=brick(list(mriTTr, mriAr))

pdf("~/SOIL-R/Manuscripts/Persistence/Figures/sattMRI.pdf")
levelplot(mribrick, names.attr=c("Mean trainsit time", "Mean age"), main="MRI")
dev.off()

pdf("~/SOIL-R/Manuscripts/Persistence/Figures/P95MRI.pdf")
levelplot(mriP95r, main="MRI")
dev.off()

###

pal=brewer.pal(10, name="Paired")

pdf("Figures/sattESMs.pdf")
plot(cesmout$meanSystemAge, cesmout$meanTransitTime, xlim=c(0,2000), ylim=c(0,2000), xlab="Mean age (years)", 
     ylab="Mean transit time (years)", pch=20, col=pal[8], bty="n")
points(ipslout$meanSystemAge, ipslout$meanTransitTime, col=pal[9], pch =20)
points(mriout$meanSystemAge, mriout$meanTransitTime, col=pal[10], pch =20)
abline(a=0,b=1, lty=2)
legend("topleft", c("CESM", "IPSL", "MRI"), pch=20, col=pal[8:10], bty="n")
dev.off()

saMeans=c(mean(cesmout$meanSystemAge), mean(ipslout$meanSystemAge), mean(mriout$meanSystemAge))
round(saMeans)
mean(saMeans)

pdf("~/SOIL-R/Manuscripts/Persistence/Figures/P95ESMs.pdf")
par(mfrow=c(3,1))
hist(cesmout$P95, main="CESM", xlab=" ")
hist(ipslout$P95, main="IPSL", xlab=" ")
hist(mriout$P95, xlim=c(0,6000), main="MRI", xlab=" Persistence P95 (years)")
par(mfrow=c(1,1))
dev.off()

saQs=c(mean(cesmout$P95), mean(ipslout$P95), mean(mriout$P95))
round(saQs)
mean(saQs)

par(mfrow=c(3,1))
hist(cesmout$meanSystemAge, main="CESM", xlab=" ", xlim=c(0,2000))
hist(ipslout$meanSystemAge, main="IPSL", xlab=" ", xlim=c(0,2000))
hist(mriout$meanSystemAge, xlim=c(0,2000), main="MRI", xlab=" Mean age (years)")
par(mfrow=c(1,1))

par(mfrow=c(3,1))
hist(cesmout$meanSystemAge/cesmout$meanTransitTime, main="CESM", xlab=" ")
hist(ipslout$meanSystemAge/ipslout$meanTransitTime, main="IPSL", xlab=" ")
hist(mriout$meanSystemAge/mriout$meanTransitTime,  main="MRI", xlab=" Mean age: mean transit time ratio")
par(mfrow=c(1,1))

ttMean=c(mean(cesmout$meanTransitTime), mean(ipslout$meanTransitTime), mean(mriout$meanTransitTime))
round(ttMean)

ttQs=c(mean(cesmout$T95), mean(ipslout$T95), mean(mriout$T95))
round(ttQs)

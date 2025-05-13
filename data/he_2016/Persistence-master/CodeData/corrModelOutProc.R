library(R.matlab)
library(raster)
library(rasterVis)
library(SoilR)
#library(sp)
library(rgdal)

#Carlos's code for persistence project
#Now also includes assessment of where we have existing incubation data

#Updated with relative paths from Persistence folder
#Alison's working directory:
setwd("C:/Users/Alison/Dropbox/14Constraint/Research/CarlosPersistenceProject/Persistence")
setwd("~/Repos/Persistence/")

# Set color bar for all maps here: ----
#Mean travel time colors:
colorBreakPts = c(0, 10,20,30,40,50,75,100, 300,500)
colorLabel = c('0', '10', '20','30','40', '50','75', '100','300','500')
#Mean age colors:
#Good for model intercomparison:
colorBreakPtsA = c(0,3000,3500,4000,4500,5000,5500,6000,9000)
colorLabelA = c('0', '3000','3500','4000','4500', '5000', '5500','6000','9000')
#Good for CESM only:
colorBreakPtsA = c(0,2000,4000,6000,8000,10000,20000)
colorLabelA = c('0', '2000','4000','6000','8000', '10000', '20000')

# Import and plot other variables (ppt & MAT) -----

# Plots of soilClim data: (10min data here, higher resolution also available)
#list of all possible variables available here: http://worldclim.org/bioclim

mat = raster("CodeData/SoilClim/wc2.0_bio_10m_01.tif")
ppt = raster("CodeData/SoilClim/wc2.0_bio_10m_12.tif")


#Plot variables of interest:
levelplot(ppt, main="Annual Precipitation")
#try hexplot:
testStack = stack(mat, ppt)
names(testStack)=c("temp","precip")
#Play with hexbinplot
#hexbinplot(temp~precip|cut(x, 6), data = testStack)
#hexbinplot(temp~precip, data = testStack)
plot(mat, ppt,  xlab="Mean Annual Temp", ylab="Mean Annual Precip", pch=20, cex=0.5, xlim=c(-50, 50), ylim=c(0, 11200))


# Extract data at test points: -------
test.dat <- structure(list(latitude = c(46.414597, 46.137664, 42.258794), longitude = c(-86.030373, -85.990492, -85.847991)), .Names = c("latitude", "longitude"), class = "data.frame", row.names = c(NA, 3L))

testPoints         <- cbind(test.dat$longitude,test.dat$latitude)
test.dat$out   <- extract(mat, testPoints)

# Extract ppt and MAT of locations where we have incubations. Plot against global distribution -----
#Read in coordinates from.txt file
coordsLitter = read.table("CodeData/testingCoordinates.txt", header = TRUE) #litter incubation coordinates
coordsSoil = read.table("CodeData/soilSiteCoordinatesSummaryfromSue.txt", header = TRUE) #soil incubation coordinates


#format points
coordsToExtractLitter = cbind(coordsLitter$Long, coordsLitter$Lat)
coordsToExtractSoil = cbind(coordsSoil$Long, coordsSoil$Lat)

#extract ppt
coordsLitter$ppt <- extract(ppt, coordsToExtractLitter)
coordsSoil$ppt <- extract(ppt, coordsToExtractSoil)

#extract MAT
coordsLitter$mat <- extract(mat, coordsToExtractLitter)
coordsSoil$mat <- extract(mat, coordsToExtractSoil)

#plot the global distribution of ppt & MAT
par(mfrow=c(1,1))
plot(mat, ppt,  xlab="Mean Annual Temp", ylab="Mean Annual Precip", pch=20, cex=0.5, col = 5, xlim=c(-50, 50), ylim=c(0, 11200))
#plot the ppt & MAT of the coordinates over top
points(coordsLitter$mat, coordsLitter$ppt,  xlab="Mean Annual Temp", ylab="Mean Annual Precip", pch=20, cex=1, col = 4, xlim=c(-50, 50), ylim=c(0, 11200))


# Zoomed in version:
plot(mat, ppt,  xlab="Mean Annual Temp", ylab="Mean Annual Precip", pch=20, cex=0.5, col = 5, xlim=c(-55, 40), ylim=c(0, 5000))
points(coordsLitter$mat, coordsLitter$ppt,  xlab="Mean Annual Temp", ylab="Mean Annual Precip", pch=20, cex=2, col = 3, xlim=c(-55, 40), ylim=c(0, 5000))
points(coordsSoil$mat, coordsSoil$ppt,  xlab="Mean Annual Temp", ylab="Mean Annual Precip", pch=20, cex=2, col = 4, xlim=c(-55, 40), ylim=c(0, 5000))
points(coordsLitter$mat, coordsLitter$ppt,  xlab="Mean Annual Temp", ylab="Mean Annual Precip", pch=20, cex=2, col = 1, xlim=c(-55, 40), ylim=c(0, 5000))


# Plots data points over the map (not working): ----
p <- levelplot(mat, main="Mean Annual Temperature")
points(testPoints)
p + layer(sp.lines(mapaSHP, lwd=0.8, col='darkgray'))

levelplot(mat, main="Mean Annual Temperature")
p + layer(sp.points(testPoints[1], testPoints[2], col=2, pch =20, cex=0.5))

#plot just test points:
plot(testPoints[,1], testPoints[,2],  xlab="Longitude", ylab="Latitude", pch=20, cex=0.5, xlim=c(-180, 180), ylim=c(-180, 180))



# Run models to get mean age and travel time ------

# CESM
cesmpar=read.table("CodeData/He/CESM/compartmentalParameters.txt", header = TRUE)
cesmout=read.csv("CodeData/He/CESM/corrCESMages.csv")

datcesm=readMat("CodeData/He/CESM/CESM_FixClim1_for3boxmodel.mat")
flcesm=datcesm$annufVegSoil
rm(datcesm) # Don't need to store this object in memory

cesmTT=matrix(NA, nrow = dim(flcesm[,,1])[1], ncol=dim(flcesm[,,1])[2]) #take row & col dimensions, NA matrix
cesmTT[cesmpar$id]=cesmout$meanTransitTime #what are the id locations?
cesmTTr=raster(cesmTT)

cesmA=matrix(NA, nrow = dim(flcesm[,,1])[1], ncol=dim(flcesm[,,1])[2])
cesmA[cesmpar$id]=cesmout$meanSystemAge
cesmAr=raster(cesmA)

cesmP95=matrix(NA, nrow = dim(flcesm[,,1])[1], ncol=dim(flcesm[,,1])[2])
cesmP95[cesmpar$id]=cesmout$P95
cesmP95r=raster(cesmP95)


#Correct coordinate system:
cesmTTr@crs #Check coordinates (currently unassigned)
#Assign coordinates to WGS84 (NO IDEA IF THIS IS NATIVE COORDS - Check Yujie's work)
# crs(cesmTTr) <- "+proj=longlat +datum=WGS84 +no_defs +ellps=WGS84 +towgs84=0,0,0"



cesmbrick=brick(list(cesmTTr, cesmAr))

pdf("Figures/corrSattCESM.pdf")
levelplot(cesmbrick, names.attr=c("Mean trainsit time", "Mean age"), main="CESM")
dev.off()

pdf("Figures/corrTTCESM.pdf")
levelplot(cesmTTr, main="Mean transit time: CESM")
dev.off()

pdf("Figures/corrSACESM.pdf")
levelplot(cesmAr, main="Mean SOM age: CESM")
dev.off()

pdf("Figures/corrP95CESM.pdf")
levelplot(cesmP95r, main="CESM")
dev.off()

#CESM maps with adjusted scalebar
levelplot(cesmTTr, main="Mean transit time: CESM", at = colorBreakPts, col.regions = terrain.colors, colorkey =list(at = colorBreakPts, labels = colorLabel))
levelplot(cesmAr, main="Mean Age: CESM", at = colorBreakPtsA, col.regions = terrain.colors, colorkey =list(at = colorBreakPtsA, labels = colorLabelA))

levelplot(cesmTTr, main="Mean transit time: CESM", at = colorBreakPts, col.regions = terrain.colors, colorkey =list(at = colorBreakPts, labels = colorLabel))
levelplot(cesmAr, main="Mean Age: CESM", at = colorBreakPtsA, col.regions = terrain.colors, colorkey =list(at = colorBreakPtsA, labels = colorLabelA))



# IPSL
ipslpar=read.table("CodeData/He/IPSL/compartmentalParameters.txt", header = TRUE)
ipslout=read.csv("CodeData/He/IPSL/corrIPSLages.csv")

datipsl=readMat("CodeData/He/IPSL/IPSL_FixClim1_for3boxmodel.mat")
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

pdf("Figures/corrSattIPSL.pdf")
levelplot(ipslbrick, names.attr=c("Mean trainsit time", "Mean age"), main="IPSL")
dev.off()

pdf("Figures/corrP95IPSL.pdf")
levelplot(ipslP95r, main="IPSL")
dev.off()

#maps with adjusted scale bars
levelplot(ipslTTr, main="Mean transit time: IPSL", at = colorBreakPts, col.regions = terrain.colors, colorkey =list(at = colorBreakPts, labels = colorLabel))
levelplot(ipslAr, main="Mean Age: IPSL", at = colorBreakPtsA, col.regions = terrain.colors, colorkey =list(at = colorBreakPtsA, labels = colorLabelA))



# MRI
mripar=read.table("CodeData/He/MRI/compartmentalParameters.txt", header = TRUE)
mriout=read.csv("CodeData/He/MRI/corrMRIages.csv")

datmri=readMat("CodeData/He/MRI/MRI_FixClim1_for3boxmodel.mat")
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

pdf("Figures/corrSattMRI.pdf")
levelplot(mribrick, names.attr=c("Mean trainsit time", "Mean age"), main="MRI")
dev.off()

pdf("Figures/corrP95MRI.pdf")
levelplot(mriP95r, main="MRI")
dev.off()

#MRI maps with adjusted scale bars
levelplot(mriTTr, main="Mean transit time: MRI", at = colorBreakPts, col.regions = terrain.colors, colorkey =list(at = colorBreakPts, labels = colorLabel))
levelplot(mriAr, main="Mean Age: MRI", at = colorBreakPtsA, col.regions = terrain.colors, colorkey =list(at = colorBreakPtsA, labels = colorLabelA))

## Hexbin plots comparing mean transit time and mean age of each model: ---------
par(mfrow=c(1,3)) #set plotting area into 2*3 array that will fill row-wise (never works on fancy plots)

pdf("Figures/hexbinplot_TTvsA_CESM.pdf")
cesmDataFrame <- data.frame(x =cesmout$meanSystemAge, y = cesmout$meanTransitTime)
hexbinplot(y~x, cesmDataFrame, aspect = 1, main="CESM", xlab="Mean Age (yrs)", ylab="Mean Transit Time (yrs)", xlim=c(0,28000), ylim=c(0,450))
dev.off()

pdf("Figures/hexbinplot_TTvsA_IPSL.pdf")
ipslDataFrame <- data.frame(x =ipslout$meanSystemAge, y = ipslout$meanTransitTime)
hexbinplot(y~x, ipslDataFrame, aspect = 1, main="IPSL", xlab="Mean Age (yrs)", ylab="Mean Transit Time (yrs)", xlim=c(0,28000), ylim=c(0,450))
dev.off()


pdf("Figures/hexbinplot_TTvsA_MRI.pdf")
mriDataFrame <- data.frame(x =mriout$meanSystemAge, y = mriout$meanTransitTime)
hexbinplot(y~x, mriDataFrame, aspect = 1, main="MRI", xlab="Mean Age (yrs)", ylab="Mean Transit Time (yrs)", xlim=c(0,28000), ylim=c(0,450))
dev.off()




par(mfrow=c(1,1)) #put back to 1*1


# Figures of mean age and transit time for the 3 different models (CESM, IPSL, MRI) ########
# comparing the 3 models - WHY doesn't this work? 
par(mfrow=c(2,3)) #set plotting area into 2*3 array that will fill row-wise
#CESM plots
levelplot(cesmTTr, main="Mean transit time: CESM", at = colorBreakPts, col.regions = terrain.colors, colorkey =list(at = colorBreakPts, labels = colorLabel))
levelplot(cesmAr, main="Mean Age: CESM", at = colorBreakPtsA, col.regions = terrain.colors, colorkey =list(at = colorBreakPtsA, labels = colorLabelA))
#IPSL plots
levelplot(ipslTTr, main="Mean transit time: IPSL", at = colorBreakPts, col.regions = terrain.colors, colorkey =list(at = colorBreakPts, labels = colorLabel))
levelplot(ipslAr, main="Mean Age: IPSL", at = colorBreakPtsA, col.regions = terrain.colors, colorkey =list(at = colorBreakPtsA, labels = colorLabelA))
#MRI plots
levelplot(mriTTr, main="Mean transit time: MRI", at = colorBreakPts, col.regions = terrain.colors, colorkey =list(at = colorBreakPts, labels = colorLabel))
levelplot(mriAr, main="Mean Age: MRI", at = colorBreakPtsA, col.regions = terrain.colors, colorkey =list(at = colorBreakPtsA, labels = colorLabelA))

# Try brick approach instead
TTbrick=brick(list(cesmTTr, ipslTTr, mriTTr)) #This doesn't work because models are not the same size?
levelplot(TTbrick, names.attr=c("CESM", "IPSL", "MRI"), main="Mean Transit Time")


#SEND TO PDF PLOTS (same as above)
#CESM plots
pdf("Figures/mapTTcolors_CESM.pdf")
levelplot(cesmTTr, main="Mean transit time: CESM", at = colorBreakPts, col.regions = terrain.colors, colorkey =list(at = colorBreakPts, labels = colorLabel))
dev.off()
pdf("Figures/mapAcolors_CESM.pdf")
levelplot(cesmAr, main="Mean Age: CESM", at = colorBreakPtsA, col.regions = terrain.colors, colorkey =list(at = colorBreakPtsA, labels = colorLabelA))
dev.off()
#IPSL plots
pdf("Figures/mapTTcolors_IPSL.pdf")
levelplot(ipslTTr, main="Mean transit time: IPSL", at = colorBreakPts, col.regions = terrain.colors, colorkey =list(at = colorBreakPts, labels = colorLabel))
dev.off()
pdf("Figures/mapAcolors_IPSL.pdf")
levelplot(ipslAr, main="Mean Age: IPSL", at = colorBreakPtsA, col.regions = terrain.colors, colorkey =list(at = colorBreakPtsA, labels = colorLabelA))
dev.off()
#MRI plots
pdf("Figures/mapTTcolors_MRI.pdf")
levelplot(mriTTr, main="Mean transit time: MRI", at = colorBreakPts, col.regions = terrain.colors, colorkey =list(at = colorBreakPts, labels = colorLabel))
dev.off()
pdf("Figures/mapAcolors_MRI.pdf")
levelplot(mriAr, main="Mean Age: MRI", at = colorBreakPtsA, col.regions = terrain.colors, colorkey =list(at = colorBreakPtsA, labels = colorLabelA))
dev.off()




# ########
pal=brewer.pal(10, name="Paired")

pdf("Figures/corrSattESMs.pdf")
plot(cesmout$meanSystemAge, cesmout$meanTransitTime, xlim=c(0,30000), ylim=c(0,1000), 
     xlab="Mean age (years)", ylab="Mean transit time (years)", pch=20, cex=0.5, col=pal[8])
points(mriout$meanSystemAge, mriout$meanTransitTime, col=pal[10], pch =20, cex=0.8)
points(ipslout$meanSystemAge, ipslout$meanTransitTime, col=pal[9], pch =20, cex=0.8)
abline(a=0,b=1, lty=2)
legend("topright", c("CESM", "IPSL", "MRI"), pch=20, col=pal[8:10], bty="n")
dev.off()

pdf("Figures/corrQ50ESMs.pdf")
plot(cesmout$P50, cesmout$T50, xlim=c(0,20000), ylim=c(0,100), 
     xlab="Median age (years)", ylab="Median transit time (years)", pch=20, cex=0.5, col=pal[8])
points(mriout$P50, mriout$T50, col=pal[10], pch =20, cex=0.8)
points(ipslout$P50, ipslout$T50, col=pal[9], pch =20, cex=0.8)
abline(a=0,b=1, lty=2)
legend("topright", c("CESM", "IPSL", "MRI"), pch=20, col=pal[8:10], bty="n")
dev.off()


pdf("Figures/corrP95ESMs.pdf")
par(mfrow=c(3,1))
hist(cesmout$P95, main="CESM", xlim=c(0,80000), xlab=" ")
hist(ipslout$P95, main="IPSL", xlim=c(0,80000), xlab=" ")
hist(mriout$P95, xlim=c(0,80000), main="MRI", xlab=" 95% quantile of age distribution (years)")
par(mfrow=c(1,1))
dev.off()

pdf("Figures/corrSAESMs.pdf")
par(mfrow=c(3,1))
hist(cesmout$meanSystemAge, main="CESM", xlab=" ", xlim=c(0,30000))
hist(ipslout$meanSystemAge, main="IPSL", xlab=" ", xlim=c(0,30000))
hist(mriout$meanSystemAge, xlim=c(0,30000), main="MRI", xlab=" Mean age (years)")
par(mfrow=c(1,1))
dev.off()

saMeans=c(mean(cesmout$meanSystemAge), mean(ipslout$meanSystemAge), mean(mriout$meanSystemAge))
round(saMeans)
mean(saMeans)

saMedians=c(mean(cesmout$P50), mean(ipslout$P50), mean(mriout$P50))
round(saMedians)

saQs=c(mean(cesmout$P95), mean(ipslout$P95), mean(mriout$P95))
round(saQs)
mean(saQs)

ttMean=c(mean(cesmout$meanTransitTime), mean(ipslout$meanTransitTime), mean(mriout$meanTransitTime))
round(ttMean)

ttMedian=c(mean(cesmout$T50), mean(ipslout$T50), mean(mriout$T50))
round(ttMedian)

ttQs=c(mean(cesmout$T95), mean(ipslout$T95), mean(mriout$T95))
round(ttQs)


# ########
Atm=bind.C14curves(prebomb=IntCal13,postbomb=Hua2013$NHZone2,time.scale="AD")

C14interp=function(x, yr=2009){ #yr=2009: Year of reference for radiocarbon curve interpolation
  Dpast=spline(Atm[,1:2], xout=x-yr)$y
  Fpast=(Dpast/1000)+1
  Ftoday=Fpast*exp(-(x)/8267)
  return((Ftoday-1)*1000)
}

Bulkcesm=C14interp(cesmout$meanSystemAge)
R14cesm=C14interp(cesmout$meanTransitTime)
Bulkipsl=C14interp(ipslout$meanSystemAge)
R14ipsl=C14interp(ipslout$meanTransitTime)
Bulkmri=C14interp(mriout$meanSystemAge)
R14mri=C14interp(mriout$meanTransitTime)
outC14=data.frame(Bulk=c(Bulkcesm, Bulkipsl, Bulkmri), R14=c(R14cesm, R14ipsl, R14mri))

plot(Bulkcesm, R14cesm,  xlab="Delta14C bulk soil", ylab="Delta14C respired CO2", pch=20, cex=0.5, xlim=c(-500, 100), ylim=c(-500, 100))
points(Bulkipsl, R14ipsl, col=2, pch =20, cex=0.5)
points(Bulkmri, R14mri, col=4, pch =20, cex=0.5)
abline(a=0,b=1, lty=2)
legend("topleft", c("CESM", "IPSL", "MRI"), pch=20, col=c(1,2,4), bty="n")

write.csv(outC14,"CodeData/modelC14.csv", row.names = FALSE)

#Calculate and Plot Global 14C of respired CO2 based on Yujie's updated model parameters

#Background info:
#Yujie's reduced complexity model uses a series 3-pool structure
#Parameters are given in text file. Loop through all parameters. Store timeseries in a matrix.
#Relevant 3-pool examples at: https://www.bgc-jena.mpg.de/TEE/basics/2015/12/30/Multiple-Pool-Radiocarbon/

#Notes and current status:
#Current file only uses CESM parameters. Could easily update to run for all models.
#Currently uses same post-bomb curve for the whole globe. Update to call correct curves for each region
#Extend forward in time to 2016 or 2017?
#Try global runs with IPSL first since it has a lower pixel resolution
#Figure out the easiest way to translate between the id coordinate system and lat/long
#Decide how to get soil properties / climate data at same locations - ask Yujie since she was further analyzing this?
#Setup to define desired model at the beginning of the code
#why is the loop saving blank png files?
#how to get the animation to run from the images?
#what is the easiest way to save R workspace variables?
#Are the constants from Yujie's model being employed correctly in soilR model setup?
#For radiocarbon movie, make sure the scale stays the same

#libraries

library(R.matlab)
library(raster)
library(rasterVis)
library(SoilR)
#library(sp)
library(rgdal)
library(magick)

#Updated with relative paths from Persistence folder
#Alison's working directory:
setwd("C:/Users/Alison/Dropbox/14Constraint/Research/CarlosPersistenceProject/Persistence")

#### MODEL INPUTS ########################################################

#Input radiocarbon curve
#Bind together pre-bomb and post-bomb (northern hemisphere zone 1 only below)
ad=bind.C14curves(prebomb=IntCal13,postbomb=Hua2013$NHZone1,time.scale="AD")
#Plot 14C curves:
par(mfrow=c(2,1))
plot(ad[,1:2],type="l", xlab="Year", ylab=expression(paste(Delta^14,"C (\u2030)")))
plot(ad[,1:2],type="l",xlim=c(0,2010), xlab="Year", ylab=expression(paste(Delta^14,"C (\u2030)")))
abline(v=1950,lty=2)
par(mfrow=c(1,1))

#Input parameters for CESM
# CESM
cesmpar=read.table("CodeData/He/CESM/compartmentalParameters.txt", header = TRUE)
cesmout=read.csv("CodeData/He/CESM/corrCESMages.csv")

datcesm=readMat("CodeData/He/CESM/CESM_FixClim1_for3boxmodel.mat")
flcesm=datcesm$annufVegSoil
rm(datcesm) # Don't need to store this object in memory

#To access transit time or mean system age (ordered by ID) use the following:
#cesmout$meanTransitTime
#cesmout$meanSystemAge
#cesmpar$id

#IPSL (lower resolution, less pixels, so easier to test things)




#### SETUP AND RUN MODEL FOR GLOBE ################################################

#Setup 3-pool series model
#Define uniform inputs:
years=seq(1940,2009,by=1) #yr
LitterInput=100 #kgC/m2/yr
k=c(k1=1/2, k2=1/10, k3=1/50) #yr^-1 (tau = yrs)
C0=c(100,500,1000) #kgC/m2
Fatm=ad[,1:2] #Mean atmospheric radiocarbon

#Define matrices data will go into:
#Each row is a timeseries for a single location. 
c14SoilMatrix = matrix(data=NA,nrow=length(cesmpar$id),ncol=length(years))
c14RespiredMatrix = matrix(data=NA,nrow=length(cesmpar$id),ncol=length(years))


#Begin looping through all parameters

startPixel = 1
endPixel = 2000 #limited number of pixels
endPixel = length(cesmpar$id) #all pixels

for(i in startPixel:endPixel) {

  #Select and assign desired parameters:
  #cesmpar has the following fields: id, tau1, tau2, tau3, rf, rs, u
  #loop through cesmpar by row, list of all relevant parameters is:
  tau1 = cesmpar[i,]$tau1
  tau2 = cesmpar[i,]$tau2 
  tau3 = cesmpar[i,]$tau3 
  rf = cesmpar[i,]$rf 
  rs = cesmpar[i,]$rs 
  u = cesmpar[i,]$u 

  #Define variable inputs
  k=c(k1=1/tau1, k2=1/tau2, k3=1/tau3) ###CHECK!? also check rf, rs, and u!
  
  a21=rf*k[1]# a21=0.9*k[1] #90% from fast to intermediate pool
  a32=rs*k[2]#a32=0.4*k[2] #40 from intermediate to slow pool

  #Define model
  Series=ThreepSeriesModel14(
    t=years,
    ks=k,
    C0=C0,
    F0_Delta14C=c(0,0,0),
    In=LitterInput,
    a21=a21,
    a32=a32,
    inputFc=Fatm
  )

  #Get output of 14C in soil C and respired C for timeseries
  # Store model output in a matrix
  #Each row is a timeseries for a given location
  #Use ID vector to figure out locations
  c14SoilMatrix[i,] = getF14C(Series) #soil C
  c14RespiredMatrix[i,]=getF14R(Series) #respired C
  #c14Pool=getF14(Series) #each individual pool C
  

  #end loop
}

#Write full matrix to file, so don't have to run it every time:
write.table(c14RespiredMatrix, file="CodeData/c14RespiredMatrix.txt", row.names=FALSE, col.names=FALSE)
write.table(c14SoilMatrix, file="CodeData/c14SoilMatrix.txt", row.names=FALSE, col.names=FALSE)
#What is the easiest way to save a workspace variable and open in another script?

save(c14RespiredMatrix, c14SoilMatrix, file = "CodeData/global14Ctimeseries.RData")
load("CodeData/global14Ctimeseries.RData")




#### RADIOCARBON TIMESERIES PLOTS ####################################################
#Select individual coordinate to plot different pools over time. 
#(just extract model output from matrix)
#(alternative option is to select parameters based on coordinates, and only run model for that site, rather than all)
j=1 #select desired row (could update to select based on id or other feature)
c14Soil = c14SoilMatrix[j,]
c14Respired = c14RespiredMatrix[j,]

#plot only respired C and mean age
par(mfrow=c(1,1))
plot(Fatm,type="l",xlab="Year",
     ylab=expression(paste(Delta^14,"C ","(\u2030)")),xlim=c(1940,2010))
lines(years,c14Soil,col=4)
lines(years,c14Respired,col=2)
legend("topright",c("Atmosphere","Bulk SOM", "Respired C"),
       lty=c(1,1,1), col=c(1,4,2),bty="n")
par(mfrow=c(1,1))


# #Loop through and plot all lines on top of that:
# for(i in startPixel:endPixel) {
#   c14Soil = c14SoilMatrix[i,]
#   c14Respired = c14RespiredMatrix[i,]
#   lines(years,c14Soil,col=4)
#   lines(years,c14Respired,col=2)
# }

#### PLOT DATA #################################################################


#### GLOBAL 14C MAPS ############################################################
#Make plots of global 14C over time (combine into a movie?)
#Loop through time. Plot global map at each time point


#Select a single particular time to plot the global map: (requires reshaping-adapt this code)
timeIndex = 70 #currently refers to the time step (column of matrix), not the absolute year. Update if desired
#reshape vector into matrix, use IDs
plotYear = years[timeIndex] #this gives you the actual year you are looking at
c14RespiredGeospatial = matrix(data = NA, dim(flcesm[,,1])[1], ncol=dim(flcesm[,,1])[2]) #take row & col dimensions, create NA matrix
c14RespiredGeospatial[cesmpar$id] = c14RespiredMatrix[,timeIndex] #fill in data using ID, only from desired time
c14RespiredGeospatialR = raster(c14RespiredGeospatial)
#levelplot(c14RespiredGeospatialR, main=expression(paste("CESM Map of ", Delta^14,"C ","(\u2030)", "  Year: ", plotYear)))
levelplot(c14RespiredGeospatialR, main=paste("CESM Radiocarbon Map,  Year: ", plotYear))

##### ANIMATION OF RADIOCARBON MAP THROUGH TIME #################################

#Initialize stack of images:
logo <- image_read("https://www.r-project.org/logo/Rlogo.png")
background1 <- image_background(image_scale(logo, "600"), "blue", flatten = TRUE) # R logo with blue background
background2 <- image_background(image_scale(logo, "600"), "white", flatten = TRUE) # R logo with white background
#imageStack = (image_morph(imageStack, 10))
imageStack <- image_scale(c(background1, background2), "600")
imageStack = image_morph(imageStack, 70) #stack initialized to have an R logo fade background color
imageStack2 = imageStack

#Plot of Just Respired C radiocarbon
for(i in 1:length(years)) {
    
  plotYear = years[i] #this gives you the actual year you are looking at
  c14RespiredGeospatial = matrix(data = NA, dim(flcesm[,,1])[1], ncol=dim(flcesm[,,1])[2]) #take row & col dimensions, create NA matrix
  c14RespiredGeospatial[cesmpar$id] = c14RespiredMatrix[,i] #fill in data using ID, only from desired time
  c14RespiredGeospatialR = raster(c14RespiredGeospatial)
  name = paste('Figures/GlobalRadiocarbonMaps/', '000',i,'plot.png',sep='')
  png(name)
    #respPlot = levelplot(c14RespiredGeospatialR, main=paste("CESM Radiocarbon Map,  Year: ", plotYear))#variable legend
    respPlot = levelplot(c14RespiredGeospatialR, main=paste("CESM Radiocarbon Map,  Year: ", plotYear), at=seq(-100,800,length.out=100)) #fixes legend to same color range for all plots

    print(respPlot)
  dev.off()
  imageStack[i] = image_read(name) #fill in imageStack with maps
}
C14animation = image_animate(imageStack,5) #plays animation if not saved to variable
image_write(C14animation, "Figures/GlobalRadiocarbonMaps/globalRespirationMapsAnimation.gif") #Save gif
print(C14animation) #Play gif in viewing window
#Examples available here: https://www.rdocumentation.org/packages/magick/versions/1.4/topics/animation
#and here: https://cran.r-project.org/web/packages/magick/vignettes/intro.html

#Plot of Soil & Respired C radiocarbon side by side
for(i in 1:length(years)) {
  
  plotYear = years[i] #this gives you the actual year you are looking at
  c14RespiredGeospatial = matrix(data = NA, dim(flcesm[,,1])[1], ncol=dim(flcesm[,,1])[2]) #take row & col dimensions, create NA matrix
  c14RespiredGeospatial[cesmpar$id] = c14RespiredMatrix[,i] #fill in data using ID, only from desired time
  c14RespiredGeospatialR = raster(c14RespiredGeospatial)
  
  c14SoilGeospatial = matrix(data = NA, dim(flcesm[,,1])[1], ncol=dim(flcesm[,,1])[2]) #take row & col dimensions, create NA matrix
  c14SoilGeospatial[cesmpar$id] = c14SoilMatrix[,i] #fill in data using ID, only from desired time
  c14SoilGeospatialR = raster(c14SoilGeospatial)
  
  c14Brick=brick(list(c14RespiredGeospatialR, c14SoilGeospatialR))
  
  name = paste('Figures/GlobalRadiocarbonMaps/', '000',i,'combinedPlot.png',sep='')
  png(name)
  respPlot = levelplot(c14Brick, main=paste("CESM Radiocarbon Map,  Year: ", plotYear), names.attr=c("Mean transit time", "Mean age"), at=seq(-100,800,length.out=100)) #fixes legend to same color range for all plots
  print(respPlot)
  dev.off()
  imageStack2[i] = image_read(name) #fill in imageStack with maps
}
C14animation2 = image_animate(imageStack2,5) #plays animation if not saved to variable
image_write(C14animation2, "Figures/GlobalRadiocarbonMaps/global14CMapsAnimation.gif") #Save gif
print(C14animation2) #Play gif in viewing window

#cesmbrick=brick(list(cesmTTr, cesmAr))
#levelplot(cesmbrick, names.attr=c("Mean transit time", "Mean age"), main="CESM")

##### GET OTHER PIXEL INFO (Lat/Long/MAT/precip etc) #############################
#Also combine with transit times or mean ages data in ID format
TT=cesmout$meanTransitTime #these are also vectors with same ID system
MeanAge=cesmout$meanSystemAge

#Convert to lat long
#Convert index to row/col, equivalent to ind2sub in Matlab (https://cran.r-project.org/doc/contrib/Hiebeler-matlabR.pdf)
id = cesmpar$id #linear index of pixel
nRowsRaster = dim(c14RespiredGeospatialR)[1]
nColRaster = dim(c14RespiredGeospatialR)[2]
pixelRowNum = ((id-1)%%nRowsRaster)+1 #Get the row for each ID if in matrix form
pixelColNum = floor((id-1)/nRowsRaster) + 1

#Calc lat long for each point:
dy = 180/nRowsRaster
dx = 360/nColRaster
lat = -(dy*pixelRowNum - dy/2) + 90 #gives lat from 90 (north) to -90 (south)
long = dx*pixelColNum - dx/2 - 180 #give long from -180(west) to 180(east)
#pixelInfo = #create data frame

#Import MAT, MAP
# Plots of soilClim data: (10min data here, higher resolution also available)
#list of all possible variables available here: http://worldclim.org/bioclim
mat = raster("CodeData/SoilClim/wc2.0_bio_10m_01.tif")
ppt = raster("CodeData/SoilClim/wc2.0_bio_10m_12.tif")
land = raster("CodeData/WorldGrids/G12IGB2a.tif") #MODIS 2012 land cover types from here: http://worldgrids.org/doku.php/wiki:layers#harmonized_world_soil_database_images_5_km
#and direct link here: http://worldgrids.org/doku.php/wiki:g12igb3
topOC = raster("CodeData/WorldGrids/TOCHWS1a.tif")#topsoil organic content: http://worldgrids.org/doku.php/wiki:tochws
soilAge = raster("CodeData/WorldGrids/geaisg3a.tif/GEAISG3a.tif") # Geological ages based on the surface geology http://worldgrids.org/doku.php/wiki:geaisg3
ET = raster("CodeData/WorldGrids/etmnts2a.tif/ETMNTS2a.tif") #evapotranspiration http://worldgrids.org/doku.php/wiki:etmnts3
plot(ET)
levelplot(soilAge)

# Extract data at test points: -------
#test.dat <- structure(list(latitude = c(46.414597, 46.137664, 42.258794), longitude = c(-86.030373, -85.990492, -85.847991)), .Names = c("latitude", "longitude"), class = "data.frame", row.names = c(NA, 3L))
#testPoints         <- cbind(test.dat$longitude,test.dat$latitude)
#test.dat$out   <- extract(mat, testPoints)

#Use lat long to extract MAT, PPT etc for each point
modelPoints = cbind(long, lat)
modelMAT = extract(mat, modelPoints)
modelMAP = extract(ppt, modelPoints)
modelLand = extract(land, modelPoints)

#plot the ppt & MAT for model points
par(mfrow=c(1,1))
plot(modelMAT, modelMAP,  xlab="Mean Annual Temp", ylab="Mean Annual Precip", pch=20, cex=0.5, col = 5, xlim=c(-50, 50), ylim=c(0, 11200))

#Store those characteristics as ID, then can use to subset the radiocarbon curves easily
dataID = data.frame(id, pixelRowNum, pixelColNum, lat, long, modelMAT, modelMAP, TT, MeanAge)

#Plot:
#ggplot(data=dataID,aes(x=modelMAT,y=modelMAP,colour=TT))+geom_point()
#ggplot(data=dataID,aes(x=modelMAT,y=modelMAP,colour=TT))+geom_point()+scale_colour_gradientn(colours = terrain.colors(10))
#ggplot(data=dataID,aes(x=modelMAT,y=modelMAP,colour=MeanAge))+geom_point()
# https://stackoverflow.com/questions/34155426/r-adjust-scale-color-gradient-in-ggplot2

ggplot(data=dataID,aes(x=modelMAT,y=modelMAP,colour=TT))+geom_point()+ scale_colour_gradientn(limits = c(0, 40), colours = terrain.colors(10)) +xlab("Mean Annual Temperature (C)") + ylab("Mean Annual Precip (mm)")+ theme_bw()
ggplot(data=dataID,aes(x=modelMAT,y=modelMAP,colour=MeanAge))+geom_point()+ scale_colour_gradientn(limits = c(0, 5000), colours = terrain.colors(10)) +xlab("Mean Annual Temperature (C)") + ylab("Mean Annual Precip (mm)")+ theme_bw()
ggplot(data=dataID,aes(x=modelMAT,y=modelMAP,colour=TT))+
  geom_point()+ scale_colour_gradient(limits = c(0, 40),low="darkblue",high="red") +
  xlab("Mean Annual Temperature (C)") + ylab("Mean Annual Precip (mm)")+ 
  theme_bw()+
  theme(panel.grid.minor = element_blank(), panel.grid.major = element_blank(),
  axis.line = element_line(colour = "black"),
  axis.title = element_text(size = rel(2)),
  axis.text = element_text(size = rel(2)))


#http://worldgrids.org/doku.php/wiki:layers#harmonized_world_soil_database_images_5_km


##### SITE SELECTION FOR RADIOCARBON CURVES ##############################

#Select subsets of different coordinates, and plot corresponding curves

#Ex: Select all sites with transit times under 15 years
#Get transit times and mean ages too, for calculations below

#Get all 14C timeseries. Plot all, or plot with mean and stdev
# Use ribbon plot: http://ggplot2.tidyverse.org/reference/geom_ribbon.html
#Do plots for different transit times

#Select sites based on their coordinates: STILL NEEDS TO BE COMPLETED
#Select sites based on MAT, MAP or biome: STILL NEEDS TO BE COMPLETED

#Select sites based on their transit times or mean ages
TT=cesmout$meanTransitTime #these are also vectors with same ID system
MeanAge=cesmout$meanSystemAge

#index = which(TT < 20) #gives indices of sites with low transit times
#index = which(TT < 50) #gives indices of sites with low transit times
index = which(TT > 100) #gives indices of sites with low transit times
#index = which(TT < 10) #gives indices of sites with low transit times
#index = which(TT < 100 & TT > 50)
#index = which(meanAge < 1000) #can also select indices based on mean soil age
index = which(modelMAT > 25)
index = which(modelMAT < -5)


# 1	L01IGB3	Evergreen needleleaf forest based on the MOD12Q1 product
# 2	L02IGB3	Evergreen broadleaf forest based on the MOD12Q1 product
# 3	L03IGB3	Deciduous needleleaf forest based on the MOD12Q1 product
# 4	L04IGB3	Deciduous broadleaf forest based on the MOD12Q1 product
# 5	L05IGB3	Mixed forests based on the MOD12Q1 product
# 6	L06IGB3	Closed shrublands based on the MOD12Q1 product
# 7	L07IGB3	Open shrublands based on the MOD12Q1 product
# 8	L08IGB3	Woody savannas forest based on the MOD12Q1 product
# 9	L09IGB3	Savannas based on the MOD12Q1 product
# 10	L10IGB3	Grasslands based on the MOD12Q1 product
# 11	L11IGB3	Permanent Wetlands based on the MOD12Q1 product
# 12	L12IGB3	Croplands based on the MOD12Q1 product
# 13	L13IGB3	Urban and built-up based on the MOD12Q1 product
# 14	L14IGB3	Cropland/natural vegetation mosaic based on the MOD12Q1 product
# 15	L15IGB3	Snow and ice based on the MOD12Q1 product
# 16	L16IGB3	Barren or sparsely vegetated based on the MOD12Q1 product
#Select by land use:
index = which(modelLand == 4) #deciduous broadleaf forest
index = which(modelLand == 10) #grasslands
index = which(modelLand == 12) #croplands


selectedSitesRespired = c14RespiredMatrix[index,]
#selectedSitesRespired = c14RespiredMatrix[1:10,] #or manually specify desired sites
selectedSitesSoil = c14SoilMatrix[index,]
#selectedSitesSoil = c14SoilMatrix[1:10,]

respiredMean = colMeans(selectedSitesRespired)
respiredSD =  apply(selectedSitesRespired, 2, sd)

soilMean = colMeans(selectedSitesSoil)
soilSD =  apply(selectedSitesSoil, 2, sd)

#Plot with ggplot ribbon
#Create a dataframe:
DataRibbon <- data.frame(x=years, yR=respiredMean, 
  lowerR = respiredMean-respiredSD, upperR = respiredMean + respiredSD,
  yS = soilMean, lowerS = soilMean-soilSD, upperS = soilMean + soilSD)

(p <-ggplot(DataRibbon) + geom_line(aes(y=yR, x=x, color = "Respired C"))+
    geom_ribbon(aes(ymin=lowerR, ymax=upperR, x=x), alpha = 0.3)+
    geom_line(aes(y=yS, x=x, color = "Soil C"))+
    geom_ribbon(aes(ymin=lowerS, ymax=upperS, x=x), alpha = 0.5)+
    geom_line(data=Fatm,aes(x=Year.AD,y=Delta14C, color = "Atmosphere"))+
    scale_colour_manual("",values=c("black", "red", "blue"))+
    scale_fill_manual("",values=c("grey12", "grey12"))+ 
    xlim(1940, 2020)+
    labs(x = "Year", y = expression(paste(Delta^14,"C ","(\u2030)")))
) # outer parentheses make plot print at the same time it is assigned





#alternative plot only respired C +/- std for grouping above
par(mfrow=c(1,1))
plot(Fatm,type="l",xlab="Year",
     ylab=expression(paste(Delta^14,"C ","(\u2030)")),xlim=c(1940,2010))
lines(years,respiredMean,col=4)
lines(years,respiredMean+respiredSD,col=2)
lines(years,respiredMean-respiredSD,col=2)
#legend("topright",c("Atmosphere","Bulk SOM", "Respired C"),
#lty=c(1,1,1), col=c(1,4,2),bty="n")
par(mfrow=c(1,1))

### FAST, MEDIUM and SLOW TT Plots #################################################
index1 = which(TT < 20) #gives indices of sites with low transit times
index2 = which(TT > 30 & TT < 50)
index3 = which(TT > 100 & TT<200 )

index1 = which(modelLand == 4) #deciduous broadleaf forest
index2 = which(modelLand == 10) #grasslands
index3 = which(modelLand == 12) #croplands

index1 = which(modelLand == 2) 
index2 = which(modelLand == 10) #grasslands
index3 = which(modelLand == 15) #snow & ice

#Also works for MAT ranges
index1 = which(dataID$modelMAT < 0) #gives indices of sites with low transit times
index2 = which(dataID$modelMAT > 5 & dataID$modelMAT < 15)
index3 = which(dataID$modelMAT > 25 )

selectedSitesRespired1 = c14RespiredMatrix[index1,]
selectedSitesSoil1 = c14SoilMatrix[index1,]
respiredMean1 = colMeans(selectedSitesRespired1)
respiredSD1 =  apply(selectedSitesRespired1, 2, sd)
soilMean1 = colMeans(selectedSitesSoil1)
soilSD1 =  apply(selectedSitesSoil1, 2, sd)

selectedSitesRespired2 = c14RespiredMatrix[index2,]
selectedSitesSoil2 = c14SoilMatrix[index2,]
respiredMean2 = colMeans(selectedSitesRespired2)
respiredSD2 =  apply(selectedSitesRespired2, 2, sd)
soilMean2 = colMeans(selectedSitesSoil2)
soilSD2 =  apply(selectedSitesSoil2, 2, sd)

selectedSitesRespired3 = c14RespiredMatrix[index3,]
selectedSitesSoil3 = c14SoilMatrix[index3,]
respiredMean3 = colMeans(selectedSitesRespired3)
respiredSD3 =  apply(selectedSitesRespired3, 2, sd)
soilMean3 = colMeans(selectedSitesSoil3)
soilSD3 =  apply(selectedSitesSoil3, 2, sd)


#Plot with ggplot ribbon
#Create a dataframe:
DataRibbon1 <- data.frame(x=years, yR1=respiredMean1, 
                         lowerR1 = respiredMean1-respiredSD1, upperR1 = respiredMean1 + respiredSD1,
                         yS1 = soilMean1, lowerS1 = soilMean1-soilSD1, upperS1 = soilMean1 + soilSD1)
DataRibbon2 <- data.frame(x=years, yR2=respiredMean2, 
                         lowerR2 = respiredMean2-respiredSD2, upperR2 = respiredMean2 + respiredSD2,
                         yS2 = soilMean2, lowerS2 = soilMean2-soilSD2, upperS2 = soilMean2 + soilSD2)
DataRibbon3 <- data.frame(x=years, yR3=respiredMean3, 
                         lowerR3 = respiredMean3-respiredSD3, upperR3 = respiredMean3 + respiredSD3,
                         yS3 = soilMean3, lowerS3 = soilMean3-soilSD3, upperS3 = soilMean3 + soilSD3)

#Plot respired C for three groups:
(p <-ggplot(DataRibbon1) + geom_line(aes(y=yR1, x=x, color = "Respired C1"))+
    geom_ribbon(aes(ymin=lowerR1, ymax=upperR1, x=x), alpha = 0.5, fill = "green")+
    
    geom_line(data = DataRibbon2, aes(y=yR2, x=x, color = "Respired C2"))+
    geom_ribbon(data = DataRibbon2, aes(ymin=lowerR2, ymax=upperR2, x=x), alpha = 0.5, fill= "#33CCCC")+
      
    geom_line(data = DataRibbon3, aes(y=yR3, x=x, color = "Respired C3"))+
    geom_ribbon(data = DataRibbon3,aes(ymin=lowerR3, ymax=upperR3, x=x), alpha = 0.5, fill= "blue")+
      
    geom_line(data=Fatm,aes(x=Year.AD,y=Delta14C, color = "Atmosphere"))+
    scale_colour_manual("",values=c("black", "darkgreen", "#33CCCC","blue"))+
   
     xlim(1940, 2020)+
    labs(x = "Year", y = expression(paste(Delta^14,"C ","(\u2030)")))+
    theme_bw() + theme(panel.grid.minor = element_blank(), panel.grid.major = element_blank())
) # outer parentheses make plot print at the same time it is assigned

#Plot soil C for three groups:
(p <-ggplot(DataRibbon1) + geom_line(aes(y=yS1, x=x, color = "Soil C1"))+
    geom_ribbon(aes(ymin=lowerS1, ymax=upperS1, x=x), alpha = 0.5, fill = "green")+
    
    geom_line(data = DataRibbon2, aes(y=yS2, x=x, color = "Soil C2"))+
    geom_ribbon(data = DataRibbon2, aes(ymin=lowerS2, ymax=upperS2, x=x), alpha = 0.5, fill= "#33CCCC")+
    
    geom_line(data = DataRibbon3, aes(y=yS3, x=x, color = "Soil C3"))+
    geom_ribbon(data = DataRibbon3,aes(ymin=lowerS3, ymax=upperS3, x=x), alpha = 0.5, fill= "blue")+
    
    geom_line(data=Fatm,aes(x=Year.AD,y=Delta14C, color = "Atmosphere"))+
    scale_colour_manual("",values=c("black", "darkgreen", "#33CCCC","blue"))+
    #scale_fill_manual("",values=c("blue", "red","green"))+ 
    xlim(1940, 2020)+
    labs(x = "Year", y = expression(paste(Delta^14,"C ","(\u2030)")))+
    theme_bw() + theme(panel.grid.minor = element_blank(), panel.grid.major = element_blank())
) # outer parentheses make plot print at the same time it is assigned

#blue/green color palette
fillColors = c("green", "#33CCCC","blue")
lineColors = c("darkgreen", "#33CCCC","blue")

#matches map color palette
fillColors = c("#CC6666", "#FFCC33","#336600")
lineColors = c("#CC6666", "#FFCC33","#336600")

fillColors = c("#FFCC33","green", "darkgreen")
lineColors = c("#FFCC33","green", "darkgreen")


#Plot soil AND respired C for three groups:
(p <-ggplot(DataRibbon1) + geom_line(aes(y=yR1, x=x, color = "Respired C1"), size = 1)+
    geom_ribbon(aes(ymin=lowerR1, ymax=upperR1, x=x), alpha = 0.5, fill = fillColors[1])+
    
    geom_line(data = DataRibbon2, aes(y=yR2, x=x, color = "Respired C2"), size = 1)+
    geom_ribbon(data = DataRibbon2, aes(ymin=lowerR2, ymax=upperR2, x=x), alpha = 0.5, fill= fillColors[2])+
    
    geom_line(data = DataRibbon3, aes(y=yR3, x=x, color = "Respired C3"), size = 1)+
    geom_ribbon(data = DataRibbon3,aes(ymin=lowerR3, ymax=upperR3, x=x), alpha = 0.5, fill= fillColors[3])+
    
    geom_line(data = DataRibbon1, aes(y=yS1, x=x, color = "Soil C1"),  linetype = "dashed")+
    geom_ribbon(data = DataRibbon1, aes(ymin=lowerS1, ymax=upperS1, x=x), alpha = 0.5, fill = fillColors[1])+
    
    geom_line(data = DataRibbon2, aes(y=yS2, x=x, color = "Soil C2"),  linetype = "dashed")+
    geom_ribbon(data = DataRibbon2, aes(ymin=lowerS2, ymax=upperS2, x=x), alpha = 0.5, fill= fillColors[2])+
    
    geom_line(data = DataRibbon3, aes(y=yS3, x=x, color = "Soil C3"), linetype = "dashed")+
    geom_ribbon(data = DataRibbon3,aes(ymin=lowerS3, ymax=upperS3, x=x), alpha = 0.5, fill= fillColors[3])+
    
    geom_line(data=Fatm,aes(x=Year.AD,y=Delta14C, color = "Atmosphere"))+
    scale_colour_manual("",values=c("black", lineColors, lineColors))+
    #scale_fill_manual("",values=c("blue", "red","green"))+ 
    xlim(1940, 2020)+
    labs(x = "Year", y = expression(paste(Delta^14,"C ","(\u2030)")))+
    theme_bw() + theme(panel.grid.minor = element_blank(), panel.grid.major = element_blank())
) # outer parentheses make plot print at the same time it is assigned
## WHY IS THERE AN ERROR ABOUT ROWS WITH MISSING VALUES?!

#Create a map identifying the same zones:

#Save as continuous variables
categoryMatrix = NA*c14RespiredGeospatial
categoryMatrix[cesmpar$id] = 0#modelMAT*0 #fill in data using ID, only from desired time
categoryMatrix[id[index1]] = 1#20
categoryMatrix[id[index2]] = 2# 80
categoryMatrix[id[index3]] = 3#200
#categoryMatrixR = raster(categoryMatrix)
categoryMatrixR = raster(x=categoryMatrix, xmn=-180,xmx = 180,ymn = -90, ymx = 90, "+proj=longlat +datum=WGS84 +no_defs +ellps=WGS84 +towgs84=0,0,0")

#Or save as factors (ordinal categorical variables)
categoryMatrix = NA*c14RespiredGeospatial
categoryMatrix[cesmpar$id] = 0 #modelMAT*0 #fill in data using ID, only from desired time
categoryMatrix[id[index1]] = 1 #20
categoryMatrix[id[index2]] = 2 # 80
categoryMatrix[id[index3]] = 3 #200
categoryMatrixR = raster(x=categoryMatrix, xmn=-180,xmx = 180,ymn = -90, ymx = 90, "+proj=longlat +datum=WGS84 +no_defs +ellps=WGS84 +towgs84=0,0,0")
categoryMatrixR = as.factor(categoryMatrixR)
#Assign raster attribute table (rat)
rat <- levels(categoryMatrixR)[[1]]
rat[["level"]] <- c("none","low", "med","high")
levels(categoryMatrixR) <- rat
levelplot(categoryMatrixR, col.regions=rev(terrain.colors(4)), xlab="", ylab="")
#or Select own colors:
my_col = c('yellow','green','blue','grey') # my_col = rev(terrain.colors(n = 4))
levelplot(categoryMatrixR, col.regions=my_col, xlab="", ylab="")
# level plot categorical here: https://stackoverflow.com/questions/19136330/legend-of-a-raster-map-with-categorical-data
#why do things always get out of order? what is the defaul ordering? 
#why does the specified color order go backwards?
#R automatically assigns factors alphabetically. figure out how to fix this


colorBreakPts = c(-1, 0, 1,2,3)
colorLabel = c('-1', '0', '1', '2','3')

colorBreakPts = c(-1, 0, 30,100,300)
colorLabel = c('-1', '0', '30', '100','300')
plot(categoryMatrixR)
levelplot(categoryMatrixR)

#levelplot(categoryMatrixR, main="Category Map", at = colorBreakPts, col.regions = topo.colors, colorkey =list(at = colorBreakPts, labels = colorLabel))
#col.regions = terrain.colors, 

##### SELECT A SINGLE SITE TO PLOT FROM LAT/LONG #############################################
#Specify coordinate(s) of interest:
#coords = cbind(long, lat)
lat = -3.018; long = -54.9714  #Tapajos
lat =-2.609	; long= -60.2091 #Manaus
lat = 55.88007; long=	98.481 #Boreas
lat = 68.42222222;	long = 149.4222222 #Toolik Lake
siteCoords =cbind(long, lat)

#Create a raster of id numbers by coordinate:
idMatrix = NA*c14RespiredGeospatial
idMatrix[cesmpar$id] = cesmpar$id #fill in data using ID, only from desired time
idMatrixR = raster(x=idMatrix, xmn=-180,xmx = 180,ymn = -90, ymx = 90, "+proj=longlat +datum=WGS84 +no_defs +ellps=WGS84 +towgs84=0,0,0")
#plot(idMatrixR) #verify it makes sense

#Extract id number of coordinate of interest
siteID = extract(idMatrixR, siteCoords)
#then find position of that within id vector
siteIDVectorIndex = which(id == siteID)

#Get soil age and respired C age for that pixel ID
#j=siteID #select desired row (could update to select based on id or other feature)
c14Soil = c14SoilMatrix[siteIDVectorIndex,]
c14Respired = c14RespiredMatrix[siteIDVectorIndex,]

#Plot curves for that pixel
#SINGLE SITE: plot only respired C and mean age
par(mfrow=c(1,1))
plot(Fatm,type="l",xlab="Year",
     ylab=expression(paste(Delta^14,"C ","(\u2030)")),xlim=c(1940,2010))
lines(years,c14Soil,col=4)
lines(years,c14Respired,col=2)
legend("topright",c("Atmosphere","Bulk SOM", "Respired C"),
       lty=c(1,1,1), col=c(1,4,2),bty="n")
par(mfrow=c(1,1))


### PLOT CURVES BASED ON LIST OF LAT/LONG

#Input longer list of coords
coordsSoil = read.table("CodeData/soilSiteCoordinatesSummaryfromSue.txt", header = TRUE) #soil incubation coordinates
siteCoords = cbind(coordsSoil$Long, coordsSoil$Lat)

#Create a raster of id numbers by coordinate:
idMatrix = NA*c14RespiredGeospatial
idMatrix[cesmpar$id] = cesmpar$id #fill in data using ID, only from desired time
idMatrixR = raster(x=idMatrix, xmn=-180,xmx = 180,ymn = -90, ymx = 90, "+proj=longlat +datum=WGS84 +no_defs +ellps=WGS84 +towgs84=0,0,0")
#plot(idMatrixR) #verify it makes sense
saveSiteID = NA*(1:length(coordsSoil$SOIL))

for(i in 1:length(coordsSoil$SOIL)) {
  
  lat = coordsSoil$Lat[i]
  long = coordsSoil$Long[i]
  siteCoords =cbind(long, lat)

  #Extract id number of coordinate of interest
  siteID = extract(idMatrixR, siteCoords)
  
  if (!is.na(siteID)) {
    #then find position of that within id vector
    siteIDVectorIndex = which(id == siteID)
    saveSiteID[i] = siteIDVectorIndex
  
    #Get soil age and respired C age for that pixel ID
    #j=siteID #select desired row (could update to select based on id or other feature)
    c14Soil = c14SoilMatrix[siteIDVectorIndex,]
    c14Respired = c14RespiredMatrix[siteIDVectorIndex,]
  
    #Plot curves for that pixel
    #SINGLE SITE: plot only respired C and mean age
    par(mfrow=c(1,1))
    plot(Fatm,type="l",xlab="Year",main= coordsSoil$SOIL[i],
        ylab=expression(paste(Delta^14,"C ","(\u2030)")),xlim=c(1940,2010))
    lines(years,c14Soil,col=4)
    lines(years,c14Respired,col=2)
    legend("topright",c("Atmosphere","Bulk SOM", "Respired C"),
          lty=c(1,1,1), col=c(1,4,2),bty="n")
    par(mfrow=c(1,1))
    
  }
}


##### CREATE HISTOGRAMS OF GLOBAL DATA ###########################################
#dataID = data.frame(id, pixelRowNum, pixelColNum, lat, long, modelMAT, modelMAP, TT, MeanAge)

ggplot(dataID, aes(TT)) +
  geom_histogram(binwidth = 5)+
  ggtitle('CESM Transit Time Histogram (All)')


p1 = ggplot(dataID, aes(TT)) +
  geom_histogram(binwidth = 5)+
  xlim(0, 200)+
  theme_bw() + theme(panel.grid.minor = element_blank(), panel.grid.major = element_blank())+
  ggtitle('CESM Transit Time Histogram (<200yrs only)')

p3 = ggplot(dataID, aes(modelMAT)) +
  geom_histogram(binwidth = 1)+
  xlim(-25,35)+
  theme_bw() + theme(panel.grid.minor = element_blank(), panel.grid.major = element_blank())+
  ggtitle('Mean Annual Temp Histogram (CESM model)')



# #subsetting options
# ggplot(subset(dataID, modelMAT>10), aes(TT))+geom_freqpoly(binwidth=5)+
#   geom_freqpoly(data = subset(dataID, modelMAT<10), aes(TT), binwidth = 5)

#### CREATE HISTOGRAMS OF SITES ####################################################


siteData = dataID[saveSiteID,]

p2 = ggplot(siteData, aes(TT)) +
  geom_histogram(binwidth = 5)+
  xlim(0, 200)+
  theme_bw() + theme(panel.grid.minor = element_blank(), panel.grid.major = element_blank())+
  ggtitle('CESM Transit Time Histogram for Data Sites')

p4 = ggplot(siteData, aes(modelMAT)) +
  geom_histogram(binwidth = 1)+
  xlim(-25,35)+
  theme_bw() + theme(panel.grid.minor = element_blank(), panel.grid.major = element_blank())+
  ggtitle('Mean Annual Temp Histogram for Data Sites')

multiplot(p1, p2, p3, p4, cols=2)



# ggplot(data=dataID,aes(x=modelMAT,y=modelMAP,colour=TT))+
#   geom_point()+ scale_colour_gradient(limits = c(0, 40),low="green",high="darkblue") +
#   xlab("Mean Annual Temperature (C)") + ylab("Mean Annual Precip (mm)")+ 
#   theme_bw()+
#   theme(panel.grid.minor = element_blank(), panel.grid.major = element_blank())

ggplot(data=dataID,aes(x=modelMAT,y=modelMAP,colour=TT))+
  geom_point()+ scale_colour_gradientn(limits = c(0, 40), colours = terrain.colors(10)) +
  geom_point(data=siteData, aes(x=modelMAT, y=modelMAP, colour=TT, size=2))+
  xlab("Mean Annual Temperature (C)") + ylab("Mean Annual Precip (mm)")+ 
  theme_bw()+
  theme(panel.grid.minor = element_blank(), panel.grid.major = element_blank())

ggplot(data=dataID,aes(x=modelMAT,y=modelMAP,colour=TT))+
  geom_point()+ scale_colour_gradientn(limits = c(0, 40), colours = terrain.colors(10)) +
  geom_point(data=siteData, aes(x=modelMAT, y=modelMAP, fill=TT), 
             colour = "black", pch=21, size =5)+
  xlab("Mean Annual Temperature (C)") + ylab("Mean Annual Precip (mm)")+ 
  theme_bw()+
  theme(panel.grid.minor = element_blank(), panel.grid.major = element_blank())

ggplot(data=dataID,aes(x=modelMAT,y=modelMAP,colour=TT))+
  geom_point()+ scale_colour_gradientn(limits = c(0, 40), colours = terrain.colors(10)) +
  geom_point(data=siteData, aes(x=modelMAT, y=modelMAP), 
             colour = "black", pch=21, size =5)+
  xlab("Mean Annual Temperature (C)") + ylab("Mean Annual Precip (mm)")+ 
  theme_bw()+ theme(panel.grid.minor = element_blank(), panel.grid.major = element_blank())

ggplot(data=dataID,aes(x=modelMAT,y=modelMAP,colour=TT))+
  geom_point()+ scale_colour_gradientn(limits = c(0, 40), colours = terrain.colors(10)) +
  geom_point(data=siteData, aes(x=modelMAT, y=modelMAP), 
             colour = "black", pch=21, size =5)+
  xlab("Mean Annual Temperature (C)") + ylab("Mean Annual Precip (mm)")+ 
  theme_bw()+ theme(panel.grid.minor = element_blank(), panel.grid.major = element_blank())

#Trying to do this:
#https://stackoverflow.com/questions/10437442/place-a-border-around-points

# ggplot(siteData,aes(x=modelMAT,y=modelMAP))+
#   geom_point(aes(fill=TT, scale_colour_gradientn(limits = c(0, 40), colours = terrain.colors(10)), colour = "black", pch=21, size =5))+ 
#   xlab("Mean Annual Temperature (C)") + ylab("Mean Annual Precip (mm)")+ 
#   theme_bw()+ theme(panel.grid.minor = element_blank(), panel.grid.major = element_blank())
# 
# g0 <- ggplot(df, aes(x=x, y=y))+geom_point(aes(fill=id), 
#                                            colour="black",pch=21, size=5))

## MAT vs MAP colored by TT
#Be careful! ppt goes to 8000, but only a few points above 6000, so cut off for viewing       
p1 = ggplot(data=dataID,aes(x=modelMAT,y=modelMAP,colour=TT))+
  geom_point()+ scale_colour_gradientn(limits = c(0, 40), colours = terrain.colors(10)) +
  xlab("Mean Annual Temperature (C)") + ylab("Mean Annual Precip (mm)")+ 
  xlim(-20,32)+ylim(0,6000)+
  theme_bw()+ theme(panel.grid.minor = element_blank(), panel.grid.major = element_blank())

p2 = ggplot(data=siteData,aes(x=modelMAT,y=modelMAP,colour=TT))+
  geom_point(size = 5)+ scale_colour_gradientn(limits = c(0, 40), colours = terrain.colors(10)) +
  xlab("Mean Annual Temperature (C)") + ylab("Mean Annual Precip (mm)")+ 
  xlim(-20,32)+ylim(0,6000)+
  theme_bw()+ theme(panel.grid.minor = element_blank(), panel.grid.major = element_blank())

multiplot(p1, p2, cols=2)


## TT vs Mean Age colored by temp
#Be careful! ppt goes to 8000, but only a few points above 6000, so cut off for viewing       
p1 = ggplot(data=dataID,aes(x=MeanAge,y=TT,colour=modelMAT))+
  geom_point()+ scale_colour_gradientn(limits = c(-25, 35), colours = terrain.colors(10)) +
  ylab("Model Transit Time (yrs)") + xlab("Model Mean Soil Age (yrs)")+   
  xlim(0,8000)+ylim(0,300)+
  theme_bw()+ theme(panel.grid.minor = element_blank(), panel.grid.major = element_blank())

p2 = ggplot(data=siteData,aes(x=MeanAge,y=TT,colour=modelMAT))+
  geom_point(size = 5)+ scale_colour_gradientn(limits = c(-25, 35), colours = terrain.colors(10)) +
  ylab("Model Transit Time (yrs)") + xlab("Model Mean Soil Age (yrs)")+ 
  xlim(0,8000)+ylim(0,300)+
  theme_bw()+ theme(panel.grid.minor = element_blank(), panel.grid.major = element_blank())


multiplot(p1, p2, cols=2)

### Plot correlations between transit time and temperature:

p1 = ggplot(data=dataID,aes(x=modelMAT,y=TT,colour=TT))+
  geom_point()+ scale_colour_gradientn(limits = c(0, 60), colours = terrain.colors(10)) +
  ylab("Model Transit Time (yrs)") + xlab("Model Mean Annual Temperature (C)")+   
  xlim(-25,35)+ylim(0,300)+
  theme_bw()+ theme(panel.grid.minor = element_blank(), panel.grid.major = element_blank())

p2 = ggplot(data=siteData,aes(x=modelMAT,y=TT,colour=TT))+
  geom_point(size = 5)+ scale_colour_gradientn(limits = c(0, 60), colours = terrain.colors(10)) +
  ylab("Model Transit Time (yrs)") + xlab("Model Mean Annual Temperature (C)")+   
  xlim(-25,35)+ylim(0,300)+
  theme_bw()+ theme(panel.grid.minor = element_blank(), panel.grid.major = element_blank())


multiplot(p1, p2, cols=2)

### Plot correlations between transit time and temperature only up to 75 years:

p1 = ggplot(data=dataID,aes(x=modelMAT,y=TT,colour=TT))+
  geom_point()+ scale_colour_gradientn(limits = c(0, 100), colours = terrain.colors(10)) +
  ylab("Model Transit Time (yrs)") + xlab("Model Mean Annual Temperature (C)")+   
  xlim(-25,35)+ylim(0,100)+
  theme_bw()+ theme(panel.grid.minor = element_blank(), panel.grid.major = element_blank())

p2 = ggplot(data=siteData,aes(x=modelMAT,y=TT,colour=TT))+
  geom_point(size = 5)+ scale_colour_gradientn(limits = c(0, 100), colours = terrain.colors(10)) +
  ylab("Model Transit Time (yrs)") + xlab("Model Mean Annual Temperature (C)")+   
  xlim(-25,35)+ylim(0,100)+
  theme_bw()+ theme(panel.grid.minor = element_blank(), panel.grid.major = element_blank())


multiplot(p1, p2, cols=2)


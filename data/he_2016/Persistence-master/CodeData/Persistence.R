library(SoilR)
library(RColorBrewer)
library(xtable)
##

#still missing data from Susan Trumbore (line 264)

#Updated with relative paths from Persistence folder
#Alison's working directory:
setwd("C:/Users/Alison/Dropbox/14Constraint/Research/CarlosPersistenceProject/Persistence")
# Carlos's working directory
setwd("~/Repos/Persistence/")


# Example 1
a=seq(1,1000)

ma=200
n=dnorm(a,mean=ma,sd=50)
ep=dexp(a,rate=1/ma)
pr=0.95
exM=qexp(p=pr,rate=1/ma)
nM=qnorm(p=pr,mean=ma,sd=50)

e50=qexp(0.5,rate=1/ma) # median of the exponential
d50=dexp(e50, rate=1/ma) # density corresponding to median


pdf("Figures/example.pdf")
plot(a,n,type="l", xlab="Soil organic matter age (years)", ylab="Probability density",col=4, bty="n")
lines(a,ep,col=2)
abline(v=ma,lty=2, col="purple", lwd=1.5)
abline(v=exM,col=2,lty=2)
text(exM+30,0.008,round(exM),col=2)
abline(v=nM,col=4,lty=2)
text(nM+30,0.008,round(nM),col=4)
segments(e50,-1,e50,d50, col=2)
dev.off()


pdf("Figures/systemTypes.pdf")
plot(c(0,10000),c(0,10000),typ="l", xlab="Ages in bulk SOM", ylab=expression(paste("Ages in release flux (",CO[2], ", DOM)")),bty="n")
polygon(c(0,10000,10000,0),c(0,0,10000,0),col=gray(0.5),border = NA)
polygon(c(0,0,10000,0),c(0,10000,10000,0),col=gray(0.8), border = NA)
lines(c(0,10000),c(0,10000), lwd=4, col=2)
legend(x=3000,y=10000, c("Type I", "Well-mixed homogeneous system"),text.col=2, bty="n")
legend(x=100,y=8000, c("Type II","Retention system"), bty="n")
legend(x=5000,y=2000, c("Type III","Non-retention system"), bty="n")
dev.off()

pdf("Figures/Palpha.pdf")
plot(a,ep,type="l", xlab="Soil organic matter age (years)", ylab="Probability density",col=2, bty="n")
segments(qexp(p=0.05,rate=1/ma),0,qexp(p=0.05,rate=1/ma),dexp(qexp(p=0.05,rate=1/ma),rate=1/ma), lty=2,col=2)
segments(qexp(p=0.5,rate=1/ma),0,qexp(p=0.5,rate=1/ma),dexp(qexp(p=0.5,rate=1/ma),rate=1/ma), lty=2,col=2)
segments(qexp(p=0.95,rate=1/ma),0,qexp(p=0.95,rate=1/ma),dexp(qexp(p=0.95,rate=1/ma),rate=1/ma), lty=2,col=2)
segments(ma,0,ma,0.003, lty=2)
abline(h=0)
legend(x=580, y=0.0005, legend=expression(Q[95]),text.col = 2, bty="n", cex=0.7)
legend(x=0, y=0.005, legend=expression(Q[5]),text.col = 2, bty="n", cex=0.7)
legend(x=110, y=0.0028, legend=expression(Q[50]),text.col = 2, bty="n", cex=0.7)
legend(x=160, y=0.0033, legend="Mean",text.col = 1, bty="n")
dev.off()

##

# RothC
ksRC=c(k.DPM=10,k.RPM=0.3,k.BIO=0.66,k.HUM=0.02)
FYMsplit=c(0.49,0.49,0.02)
DR=1.44; In=1.7; FYM=0; clay=23.4
x=1.67*(1.85+1.60*exp(-0.0786*clay))
B=0.46/(x+1) # Proportion that goes to the BIO pool
H=0.54/(x+1) # Proportion that goes to the HUM pool

ai3=B*ksRC
ai4=H*ksRC

ARC=diag(-ksRC)
ARC[3,]=ARC[3,]+ai3
ARC[4,]=ARC[4,]+ai4

RcI=matrix(nrow=4,ncol=1,c(In*(DR/(DR+1))+(FYM*FYMsplit[1]),In*(1/(DR+1))+(FYM*FYMsplit[2]),0,(FYM*FYMsplit[3])))

tau=seq(1,1000); qs=seq(0.05, 0.95, by=0.05)
aRC=systemAge(A=ARC,u=RcI,a=tau, q=qs)
tRC=transitTime(A=ARC,u=RcI,a=tau, q=qs)

# Century
ksC=c(k.STR=0.094,k.MET=0.35,k.ACT=0.14,k.SLW=0.0038,k.PAS=0.00013)
clayC=0.2; silt=0.45; LN=0.5; Ls=0.1; In=0.1
Txtr=clayC+silt
fTxtr=1-0.75*Txtr
Es=0.85-0.68*Txtr

alpha31=0.45; alpha32=0.55; alpha34=0.42; alpha35=0.45
alpha41=0.3; alpha53=0.004; alpha43=1-Es-alpha53; alpha54=0.03

AC=-1*diag(abs(ksC))
AC[1,1]=AC[1,1]*exp(-3*Ls)
AC[3,3]=AC[3,3]*fTxtr
AC[3,1]=alpha31*abs(AC[1,1])
AC[3,2]=alpha32*abs(AC[2,2])
AC[3,4]=alpha34*abs(AC[4,4])
AC[3,5]=alpha35*abs(AC[5,5])
AC[4,1]=alpha41*abs(AC[1,1])
AC[4,3]=alpha43*abs(AC[3,3])
AC[5,3]=alpha53*abs(AC[3,3])
AC[5,4]=alpha54*abs(AC[4,4])

Fm=0.85-0.18*LN; Fs=1-Fm
CI=matrix(nrow=5,ncol=1,c(In*Fm,In*Fs,0,0,0))

aC=systemAge(A=AC, u=CI, a=tau, q=qs)
tC=transitTime(A=AC, u=CI, a=tau, q=qs)


# Yasso07
ksY=c(kA=0.66, kW=4.3, kE=0.35, kN=0.22, kH=0.0033)
p=c(p1=0.32,p2=0.01,p3=0.93,p4=0.34,p5=0,p6=0,p7=0.035,p8=0.005,p9=0.01,p10=0.0005,p11=0.03,p12=0.92,pH=0.04)
A1=abs(diag(ksY))
Ap=diag(-1,5,5)
Ap[1,2]=p[1]; Ap[1,3]=p[2]; Ap[1,4]=p[3]; Ap[2,1]=p[4]; Ap[2,3]=p[5]; Ap[2,4]=p[6]
Ap[3,1]=p[7]; Ap[3,2]=p[8]; Ap[3,4]=p[9]; Ap[4,1]=p[10]; Ap[4,2]=p[11]; Ap[4,3]=p[12]; Ap[5,1:4]=p[13]

AY=Ap%*%A1
IY=matrix(nrow=5,ncol=1,c(10,0,0,0,0))

aY=systemAge(A=AY, u=IY,a=tau, q=qs)
tY=transitTime(A=AY, u=IY,a=tau, q=qs)


# CLM4cn
Une=matrix(c(299,84,200), 3, 1) # Needleleaf evergreen (same for temperate and boreal) - CLM PFTs as in Wieder et al. 2014
Ubdt=matrix(c(399,23,180),3,1) # Values for Broadleaf deciduous temperate tree 
Ubet=matrix(c(910,539,360), 3, 1) # Broadleaf evergreen tropical
B=matrix(c(0.25,0.5,0.25,rep(0,5),
           0.25,0.5,0.25,rep(0,5),
           rep(0,3),1,rep(0,4)),8,3)

uCLM1=B%*%Une
uCLM2=B%*%Ubdt
uCLM3=B%*%Ubet

K=diag(c(434.0,26.47,5.145,0.3652,26.47,5.145,0.5114,0.0365))
A=diag(-1,8)
A[5,1]=0.61; A[6,2]=0.45; A[7,3]=0.71; A[2,4]=0.76; A[3,4]=0.24; A[6,5]=0.72; A[7,6]=0.54; A[8,7]=0.45

ACLM=A%*%K

aCLM1=systemAge(A=ACLM,u=uCLM1,a=tau, q=qs)
tCLM1=transitTime(A=ACLM,u=uCLM1,a=tau, q=qs)
aCLM2=systemAge(A=ACLM,u=uCLM2,a=tau, q=qs)
tCLM2=transitTime(A=ACLM,u=uCLM2,a=tau, q=qs)
aCLM3=systemAge(A=ACLM,u=uCLM3,a=tau, q=qs)
tCLM3=transitTime(A=ACLM,u=uCLM3,a=tau, q=qs)

#ICBM model for the -N, -straw treatment, interpreted as control
uICBM=matrix(c(0.057,0),2,1)
AICBM=matrix(c(-0.936, 0, 
               0.117, -0.0070785), 2,2, byrow=TRUE)

aICBM=systemAge(A=AICBM,u=uICBM, a=tau, q=qs)
tICBM=transitTime(A=AICBM, u=uICBM, a=tau, q=qs)

## Models from He et al. 2016
#CESM
cesmpar=read.table("CodeData/He/CESM/compartmentalParameters.txt", header = TRUE)
cesmMeanpar=apply(cesmpar,MARGIN = 2, FUN=mean)
Acesm=diag(-1/cesmMeanpar[2:4])
Acesm[3,3]=-1/(cesmMeanpar[4]*3.7)
Acesm[2,1]=cesmMeanpar[5]/cesmMeanpar[2]
Acesm[3,2]=0.34*cesmMeanpar[6]/cesmMeanpar[3]
ucesm = matrix(c(cesmMeanpar[7],0,0),3,1)

aCESM=systemAge(A=Acesm, u=ucesm, a=tau, q=qs)
tCESM=transitTime(A=Acesm, u=ucesm, a=tau, q=qs)

#IPSL
ipslpar=read.table("CodeData/He/IPSL/compartmentalParameters.txt", header = TRUE)
ipslMeanpar=apply(ipslpar,MARGIN = 2, FUN=mean)
Aipsl=diag(-1/ipslMeanpar[2:4])
Aipsl[3,3]=-1/(ipslMeanpar[4]*14)
Aipsl[2,1]=ipslMeanpar[5]/ipslMeanpar[2]
Aipsl[3,2]=0.07*ipslMeanpar[6]/ipslMeanpar[3]
uipsl = matrix(c(ipslMeanpar[7],0,0),3,1)

aIPSL=systemAge(A=Aipsl, u=uipsl, a=tau, q=qs)
tIPSL=transitTime(A=Aipsl, u=uipsl, a=tau, q=qs)

#MRI
mripar=read.table("CodeData/He/MRI/compartmentalParameters.txt", header = TRUE)
mriMeanpar=apply(mripar,MARGIN = 2, FUN=mean)
Amri=diag(-1/mriMeanpar[2:4])
Amri[3,3]=-1/(mriMeanpar[4]*13)
Amri[2,1]=0.46*mriMeanpar[5]/mriMeanpar[2]
Amri[3,2]=0.34*mriMeanpar[6]/mriMeanpar[3]
umri = matrix(c(mriMeanpar[7],0,0),3,1)

aMRI=systemAge(A=Amri, u=umri, a=tau, q=qs)
tMRI=transitTime(A=Amri, u=umri, a=tau, q=qs)

ageModels=list(aRC, aC, aY, aICBM, aCLM1, aCLM2, aCLM3, aCESM, aIPSL, aMRI) #List of all output
ttModels=list(tRC, tC, tY, tICBM, tCLM1, tCLM2, tCLM3, tCESM, tIPSL, tMRI) #List of all output

listSubset=function(x,name){x[name]}

meanAges=unlist(sapply(X=ageModels, FUN=listSubset, name="meanSystemAge"), use.names = FALSE)
meantransitTimes=unlist(sapply(X=ttModels, FUN=listSubset, name="meanTransitTime"), use.names = FALSE)
QA=sapply(X=ageModels, FUN=listSubset, name="quantilesSystemAge")
QT=sapply(X=ttModels, FUN=listSubset, name="quantiles")
Aq95=sapply(QA, max)
Aq50=sapply(QA, median)
Tq95=sapply(QT, max)
Tq50=sapply(QT, median)

# meanAges=c(aRC$meanSystemAge, aC$meanSystemAge, aY$meanSystemAge, aCLM1$meanSystemAge, aCLM2$meanSystemAge, aCLM3$meanSystemAge, aICBM$meanSystemAge, aCESM$meanSystemAge, aIPSL$meanSystemAge, aMRI$meanSystemAge)
# meantransitTimes=c(tRC$meanTransitTime, tC$meanTransitTime, tY$meanTransitTime, tCLM1$meanTransitTime, tCLM2$meanTransitTime, tCLM3$meanTransitTime, tICBM$meanTransitTime, tCESM$meanTransitTime, tIPSL$meanTransitTime, tMRI$meanTransitTime)
# P95=c(aRC$quantilesSystemAge[3], aC$quantilesSystemAge[3], aY$quantilesSystemAge[3], aCLM1$quantilesSystemAge[3], aCLM2$quantilesSystemAge[3], aCLM3$quantilesSystemAge[3], aICBM$quantilesSystemAge[3], aCESM$quantilesSystemAge[3], aIPSL$quantilesSystemAge[3], aMRI$quantilesSystemAge[3])
# PT95=c(tRC$quantiles[3], tC$quantiles[3], tY$quantiles[3], tCLM1$quantiles[3], tCLM2$quantiles[3], tCLM3$quantiles[3], tICBM$quantiles[3], tCESM$quantiles[3], tIPSL$quantiles[3], tMRI$quantiles[3])
# P5=c(aRC$quantilesSystemAge[1], aC$quantilesSystemAge[1], aY$quantilesSystemAge[1], aCLM1$quantilesSystemAge[1], aCLM2$quantilesSystemAge[1], aCLM3$quantilesSystemAge[1], aICBM$quantilesSystemAge[1], aCESM$quantilesSystemAge[1], aIPSL$quantilesSystemAge[1], aMRI$quantilesSystemAge[1])
# PT5=c(tRC$quantiles[1], tC$quantiles[1], tY$quantiles[1], tCLM1$quantiles[1], tCLM2$quantiles[1], tCLM3$quantiles[1], tICBM$quantiles[1], tCESM$quantiles[1], tIPSL$quantiles[1], tMRI$quantiles[1])

pal=brewer.pal(10, name="Paired")

modelNames=c("RothC", "Century", "Yasso07", "ICBM", "CLM4cn-Needleleaf", "CLM4cn-Deciduous", "CLM4cn-Tropical", "CESM", "IPSL", "MRI")

pdf("Figures/modelsSA.pdf")
par(mar=c(5,4,1,2))
plot(tau,aRC$systemAgeDensity,type="l", ylim=c(0,0.06), xlim=c(0,200), ylab="Probability density function", xlab="Age (years)", col=pal[1], bty="n")
lines(tau,aC$systemAgeDensity,col=pal[2])
lines(tau,aY$systemAgeDensity,col=pal[3])
lines(tau,aICBM$systemAgeDensity,col=pal[4])
lines(tau,aCLM1$systemAgeDensity,col=pal[5])
lines(tau,aCLM2$systemAgeDensity,col=pal[6])
lines(tau,aCLM3$systemAgeDensity,col=pal[7])
lines(tau,aCESM$systemAgeDensity,col=pal[8])
lines(tau,aIPSL$systemAgeDensity,col=pal[9])
lines(tau,aMRI$systemAgeDensity,col=pal[10])
#abline(v=meanAges, lty=2, col=pal)
legend("topright", modelNames, col=pal, lty=1, bty="n")
dev.off()

pdf("Figures/modelsTT.pdf")
par(mar=c(5,4,1,2))
plot(tau,tRC$transitTimeDensity,type="l", ylim=c(0,0.15), xlim=c(0,100), ylab="Probability density function", xlab="Transit time (years)", col=pal[1], bty="n")
lines(tau,tC$transitTimeDensity,col=pal[2])
lines(tau,tY$transitTimeDensity,col=pal[3])
lines(tau,tICBM$transitTimeDensity,col=pal[4])
lines(tau,tCLM1$transitTimeDensity,col=pal[5])
lines(tau,tCLM2$transitTimeDensity,col=pal[6])
lines(tau,tCLM3$transitTimeDensity,col=pal[7])
lines(tau,tCESM$transitTimeDensity,col=pal[8])
lines(tau,tIPSL$transitTimeDensity,col=pal[9])
lines(tau,tMRI$transitTimeDensity,col=pal[10])
legend("topright", modelNames, col=pal, lty=1, bty="n")
dev.off()

pdf("Figures/ModelQ.pdf")
plot(QA[[1]],QT[[1]], type="b", col=pal[1], ylim=c(0.001,1000),xlim=c(0.001,30000), log="xy", xlab="Age quantiles (years)", 
     ylab="Transit time quantiles (years)", pch=20, cex=0.8, bty="n")
for(i in 2:10){
 points(QA[[i]], QT[[i]], type="b", col=pal[i], pch=20, cex=0.8)
}
abline(0,1,lty=2)
legend("topleft", modelNames, col=pal, pch=20,lty=1, bty="n")
dev.off()

tabQ95=xtable(data.frame(Model=modelNames,Mean.Age=round(meanAges, 1), Age50=round(Aq50, 1),Age95=round(Aq95, 1),MeanTransitTime=round(meantransitTimes, 1),
                         Q50=round(Tq50, 1),Q95=round(Tq95, 1)),
               caption="Mean, 50, and 95 percent quantiles of the age and transit time distributions for all models", label="tab:Qs")
print(tabQ95, file="tabQ95.tex", include.rownames=FALSE, caption.placement="top", booktabs=TRUE)

#xtable(data.frame(Model=modelNames,MeanTransitTime=round(meantransitTimes, 1),Q95=round(Tq95, 1)), include.rownames=FALSE)

###########################################################################################################################
# Print matrices for appendix

# RothC
matARC=xtable(ARC,align=rep("",ncol(ARC)+1))
uRC=xtable(RcI,align=rep("",2))
print(matARC,file="BrothC.tex",
      floating=FALSE, tabular.environment="pmatrix", hline.after=NULL, include.rownames=FALSE, include.colnames=FALSE)
print(uRC,file="uRothC.tex",
      floating=FALSE, tabular.environment="pmatrix", hline.after=NULL, include.rownames=FALSE, include.colnames=FALSE)

# Century
matC=xtable(round(AC, 6),align=rep("",ncol(ARC)+1), digits=6)
uC=xtable(CI,align=rep("",2))
print(matC,file="Bcentury.tex",
      floating=FALSE, tabular.environment="pmatrix", hline.after=NULL, include.rownames=FALSE, include.colnames=FALSE)
print(uC,file="uCentury.tex",
      floating=FALSE, tabular.environment="pmatrix", hline.after=NULL, include.rownames=FALSE, include.colnames=FALSE)

# Yasso07
matY=xtable(round(AY, 4),align=rep("",ncol(AY)+1), digits=4)
uY=xtable(IY,align=rep("",2))
print(matY,file="Byasso.tex",
      floating=FALSE, tabular.environment="pmatrix", hline.after=NULL, include.rownames=FALSE, include.colnames=FALSE)
print(uY,file="uYasso.tex",
      floating=FALSE, tabular.environment="pmatrix", hline.after=NULL, include.rownames=FALSE, include.colnames=FALSE)

# ICBM
matICBM=xtable(round(AICBM, 4),align=rep("",ncol(AICBM)+1), digits=4)
uIcbm=xtable(uICBM,align=rep("",2))
print(matICBM,file="Bicbm.tex",
      floating=FALSE, tabular.environment="pmatrix", hline.after=NULL, include.rownames=FALSE, include.colnames=FALSE)
print(uIcbm,file="uICBM.tex",
      floating=FALSE, tabular.environment="pmatrix", hline.after=NULL, include.rownames=FALSE, include.colnames=FALSE)

# CLM
matCLM=xtable(round(ACLM, 4),align=rep("",ncol(ACLM)+1), digits=4)
uclm1=xtable(uCLM1,align=rep("",2))
uclm2=xtable(uCLM2,align=rep("",2))
uclm3=xtable(uCLM3,align=rep("",2))
print(matCLM,file="Bclm.tex",
      floating=FALSE, tabular.environment="pmatrix", hline.after=NULL, include.rownames=FALSE, include.colnames=FALSE)
print(uclm1,file="uCLM1.tex",
      floating=FALSE, tabular.environment="pmatrix", hline.after=NULL, include.rownames=FALSE, include.colnames=FALSE)
print(uclm2,file="uCLM2.tex",
      floating=FALSE, tabular.environment="pmatrix", hline.after=NULL, include.rownames=FALSE, include.colnames=FALSE)
print(uclm3,file="uCLM3.tex",
      floating=FALSE, tabular.environment="pmatrix", hline.after=NULL, include.rownames=FALSE, include.colnames=FALSE)

# CESM
matCESM=xtable(round(Acesm, 4),align=rep("",ncol(Acesm)+1), digits=4)
uCESM=xtable(ucesm,align=rep("",2))
print(matCESM,file="Bcesm.tex",
      floating=FALSE, tabular.environment="pmatrix", hline.after=NULL, include.rownames=FALSE, include.colnames=FALSE)
print(uCESM,file="uCESM.tex",
      floating=FALSE, tabular.environment="pmatrix", hline.after=NULL, include.rownames=FALSE, include.colnames=FALSE)

# IPSL
matIPSL=xtable(round(Aipsl, 5),align=rep("",ncol(Aipsl)+1), digits=5)
uIPSL=xtable(uipsl,align=rep("",2))
print(matIPSL,file="Bipsl.tex",
      floating=FALSE, tabular.environment="pmatrix", hline.after=NULL, include.rownames=FALSE, include.colnames=FALSE)
print(uIPSL,file="uIPSL.tex",
      floating=FALSE, tabular.environment="pmatrix", hline.after=NULL, include.rownames=FALSE, include.colnames=FALSE)

# MRI
matMRI=xtable(round(Amri, 5),align=rep("",ncol(Amri)+1), digits=5)
uMRI=xtable(umri,align=rep("",2))
print(matMRI,file="Bmri.tex",
      floating=FALSE, tabular.environment="pmatrix", hline.after=NULL, include.rownames=FALSE, include.colnames=FALSE)
print(uMRI,file="uMRI.tex",
      floating=FALSE, tabular.environment="pmatrix", hline.after=NULL, include.rownames=FALSE, include.colnames=FALSE)

###########################################################################################################################
##### Radiocarbon data
#Data from Ingo Schoening
EPbulk=read.csv("CodeData/14C_SOM.csv")
EPresp=read.csv("CodeData/14C_Respiration.csv")
EPcn=read.csv("CodeData/CN_Exploratories.csv")

Datm=40 # Summer Delta14C at Schauinsland. Data from I. Levin
k=match(EPbulk[,1],EPresp[,1])
EP=data.frame(EPbulk,R14=EPresp[k,6])

ICplots=EPcn[which(EPcn$Inorganic_C/EPcn$Total_C >= 0.01),1] # Plots with inorganic C > 1%
EPclean=EP[-na.exclude(match(ICplots, EP[,1])),]


# Data from Susan Trumbore
c14 <- read.csv("~/SOIL-R/Manuscripts/Mean_Age_and_Transit_Time/Opinion/RadiocarbonData.csv")
c14[,12]=c14$C14.mineralSOM-c14$C14.Atmosphere  ## delta14C mineral soil - delta 14C atmosphere in year of sampling
c14[,13]=c14$C14.MineralCO2-c14$C14.Atmosphere ## delta14C CO2 respired from mineral soil - delta 14C atmosphere in year of sampling

# Data from Karis McFarlene
McF=read.csv("CodeData/KMcF.csv")

litter=rbind(as.matrix(c14[,6:7]),
         as.matrix(McF[McF[,4]=="Litterfall",6:7]))

OH=rbind(McF[McF[,4]=="Oi", 6:7],
         McF[McF[,4]=="O horizon", 6:7])

top=rbind(as.matrix(c14[,12:13]),
          as.matrix(McF[McF[,4]=="0-5 cm",6:7]),
          as.matrix(McF[McF[,4]=="5-10 cm",6:7]),
          as.matrix(McF[McF[,4]=="10-15 cm", 6:7]),
          as.matrix(EPclean[,5:6]))

subsoil=rbind(McF[McF[,4]=="15.5-20.5 cm", 6:7],
              McF[McF[,4]=="23.5-28.5 cm", 6:7],
              McF[McF[,4]=="38.5-43.5 cm", 6:7],
              McF[McF[,4]=="41.5-45.5 cm", 6:7],
              McF[McF[,4]=="72.5-77.5 cm", 6:7])

modelC14=read.csv("CodeData/modelC14.csv")

pdfFonts(stdsym=Type1Font("standardsymbol",
                          c("Helvetica.afm",
                            "s050000l.afm",
                            "Helvetica-Oblique.afm",
                            "Helvetica-BoldOblique.afm",
                            "Symbol.afm"),
                          encoding="UTF-8"))

pal2=c("#d11141", "#00b159", "#00aedb", "#f37735")
pdf("Figures/incubations.pdf")
plot(modelC14, xlim=c(-500,200), ylim=c(-500,200), bty="n", col="gray",
     xlab=expression(paste(Delta^14, "C in organic matter (", "\U2030", ")")),
     ylab=expression(paste(Delta^14, "C in respired C", O[2], " (\U2030)")), pch=20, cex=0.3)
points(litter,pch=20, col=pal2[1])
points(OH, pch=20, col=pal2[2])
points(top,pch=20, col=pal2[3])
points(subsoil, pch=20, col=pal2[4])
abline(h=0,lty=2)
abline(v=0,lty=2)
abline(a=0,b=1,lty=2)
legend("bottomright", c("Litter", "O horizon", "Mineral topsoil (0-15 cm)", "Subsoil (below 20 cm)", "ESMs prediction"),
       bty="n",  cex=1.2, pch= 20, col = c(pal2[1:4], "gray"))
dev.off()

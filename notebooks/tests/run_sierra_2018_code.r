library(SoilR)


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

tau=seq(1,1000)
aRC=systemAge(A=ARC,u=RcI,a=tau )
age_pdf = aRC$systemAgeDensity
write.table(cbind(tau,age_pdf),file="sierra_2018_RothC.csv",sep=",",row.names=FALSE)
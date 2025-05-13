
#This script calculates mean ages and transit times as well as quantiles for output from the MRI model with corrected values from He's paper

library(SoilR)

mri=read.table("~/SOIL-R/Manuscripts/Persistence/CodeData/He/MRI/compartmentalParameters.txt", header = TRUE)

out=NULL
for(i in 1:nrow(mri)){
  A=diag(-1/mri[i,2:4])
  A[3,3]=-1/(mri[i,4]*13) #corrected
  A[2,1]=0.46*mri[i,5]/mri[i,2] #corrected
  A[3,2]=0.34*mri[i,6]/mri[i,3] #corrected
  u = matrix(c(mri[i,7],0,0),3,1)
  
  age=systemAge(A=A, u=u, a=1)
  trt=transitTime(A=A, u=u, a=1)
  rows=c(trt$meanTransitTime, trt$quantiles, age$meanSystemAge, age$quantilesSystemAge)
  out=rbind(out,rows)
  print(i)
}

outdf=data.frame(out)
names(outdf)<-c("meanTransitTime", "T5", "T50", "T95", "meanSystemAge", "P5", "P50", "P95")
write.csv(outdf,"~/SOIL-R/Manuscripts/Persistence/CodeData/He/MRI/corrMRIages.csv", row.names = FALSE )
print("Finished succesfully!")

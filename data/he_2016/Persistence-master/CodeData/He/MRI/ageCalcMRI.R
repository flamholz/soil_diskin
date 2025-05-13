
#This script calculates mean ages and transit times as well as quantiles for output from the MRI model

library(SoilR)

mri=read.table("~/SOIL-R/Manuscripts/Persistence/CodeData/He/MRI/compartmentalParameters.txt", header = TRUE)

out=NULL
for(i in 1:nrow(mri)){
  A=diag(-1/mri[i,2:4])
  A[2,1]=mri[i,5]/mri[i,2]
  A[3,2]=mri[i,6]/mri[i,3]
  u = matrix(c(mri[i,7],0,0),3,1)
  
  age=systemAge(A=A, u=u, a=1)
  trt=transitTime(A=A, u=u, a=1)
  rows=c(trt$meanTransitTime, trt$quantiles, age$meanSystemAge, age$quantilesSystemAge)
  out=rbind(out,rows)
  print(i)
}

outdf=data.frame(out)
names(outdf)<-c("meanTransitTime", "T5", "T50", "T95", "meanSystemAge", "P5", "P50", "P95")
write.csv(outdf,"~/SOIL-R/Manuscripts/Persistence/CodeData/He/MRI/MRIages.csv", row.names = FALSE )
print("Finished succesfully!")

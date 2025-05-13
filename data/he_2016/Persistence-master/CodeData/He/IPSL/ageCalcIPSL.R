
#This script calculates mean ages and transit times as well as quantiles for output from the IPSL model

library(SoilR)

ipsl=read.table("~/SOIL-R/Manuscripts/Persistence/CodeData/He/IPSL/compartmentalParameters.txt", header = TRUE)

out=NULL
for(i in 1:nrow(ipsl)){
  A=diag(-1/ipsl[i,2:4])
  A[2,1]=ipsl[i,5]/ipsl[i,2]
  A[3,2]=ipsl[i,6]/ipsl[i,3]
  u = matrix(c(ipsl[i,7],0,0),3,1)
  
  age=systemAge(A=A, u=u, a=1)
  trt=transitTime(A=A, u=u, a=1)
  rows=c(trt$meanTransitTime, trt$quantiles, age$meanSystemAge, age$quantilesSystemAge)
  out=rbind(out,rows)
  print(i)
}

outdf=data.frame(out)
names(outdf)<-c("meanTransitTime", "T5", "T50", "T95", "meanSystemAge", "P5", "P50", "P95")
write.csv(outdf,"~/SOIL-R/Manuscripts/Persistence/CodeData/He/IPSL/IPSLages.csv", row.names = FALSE )
print("Finished succesfully!")


#This script calculates mean ages and transit times as well as quantiles for output from the CESM model using correction factors from He's paper

library(SoilR)

cesm=read.table("~/SOIL-R/Manuscripts/Persistence/CodeData/He/CESM/compartmentalParameters.txt", header = TRUE)

out=NULL
for(i in 1:nrow(cesm)){
  A=diag(-1/cesm[i,2:4])
  A[3,3]=-1/(cesm[i,4]*3.7) #corrected rate
  A[2,1]=cesm[i,5]/cesm[i,2]
  A[3,2]=0.34*cesm[i,6]/cesm[i,3] #corrected transfer coefficient
  u = matrix(c(cesm[i,7],0,0),3,1)
  
  age=systemAge(A=A, u=u, a=1)
  trt=transitTime(A=A, u=u, a=1)
  rows=c(trt$meanTransitTime, trt$quantiles, age$meanSystemAge, age$quantilesSystemAge)
  out=rbind(out,rows)
  print(i)
}

outdf=data.frame(out)
names(outdf)<-c("meanTransitTime", "T5", "T50", "T95", "meanSystemAge", "P5", "P50", "P95")
write.csv(outdf,"~/SOIL-R/Manuscripts/Persistence/CodeData/He/CESM/corrCESMages.csv", row.names = FALSE )
print("Finished succesfully!")

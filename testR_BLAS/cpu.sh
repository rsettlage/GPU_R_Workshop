Rscript -e "N<-c(25,50,500,1000,5000,7500,10000,12500); for(n in seq_along(N)){A<-matrix(1:N[n]*2,nrow=N[n],ncol=N[n]);time1<-system.time(temp<-t(A)%*%A); cat('nrow=ncol=',N[n],' time ',time1[3],'\n');}"


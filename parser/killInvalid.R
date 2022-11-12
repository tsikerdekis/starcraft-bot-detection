library(dplyr)
killinvalid <- function(fn) {
    csv <- read.csv(fn)
    frame <- as.integer(csv[1,1])
 if(!is.na(frame)) {
 if(frame > 360) {
    file.remove(fn)
     print(fn)
    }
    }
}





filesh <- list.files("gamefiles/csvs/15k/human", pattern="*.csv", full.names=TRUE)
filesb <- list.files("gamefiles/csvs/15k/bot", pattern="*.csv", full.names=TRUE)
lapply(filesh, killinvalid)
lapply(filesb, killinvalid)

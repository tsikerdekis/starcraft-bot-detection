library(dplyr)

human <- function(filename) {
  cut(filename, "human", 1)
}

bot <- function(filename) {
  cut(filename, "bot", 0)
}

cut <- function(filename, bh, type) {
  csv <- read.csv(filename)
  out <- csv %>% filter(Frame < 15000)
  local = ""
  if(type == 1) {
    local = sub(bh, "15k/human", gsub(" ", "", filename))
  }
  else {
    local = sub(bh, "15k/bot", gsub(" ", "", filename))
  }
  write.csv(out, local, row.names = F)
}





 filesh <- list.files("gamefiles/csvs/human", pattern="*.csv", full.names=TRUE)
 filesb <- list.files("gamefiles/csvs/bot", pattern="*.csv", full.names=TRUE)
 lapply(filesh, human)
 lapply(filesb, bot)

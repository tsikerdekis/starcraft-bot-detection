library(jsonlite)
library(plyr)
library(stringr)
library(data.table)
library(dplyr)
library(pracma)

game <- 1

humanCSV <- function(filename) {
  makeCSV(filename, "h")
}

botCSV <- function(filename) {
  makeCSV(filename, "b")
}

na.neg <- function(x) {
  x[is.na(x)] <- -1
  x[is.null(x)] <- -1
  return(x)
}



makeCSV <- function(filename, borh) {
  JsonData <- fromJSON(filename)
  data <- JsonData[2]
  data <- data[1]
  data2 <- unlist(data, recursive=FALSE)
  df <- data.frame(data2[1])
  players <- split(df, df$Commands.Cmds.PlayerID)
  player <- 1
  for (i in seq_along(players)) {
  player1 <- flatten(as.data.frame(players[i]))
  sample1 <- player1
  locale <- ""
  if(strcmp(borh, "b")) {
    locale <- "gamefiles/csvs/bot/"
  }
  else {
    locale <- "gamefiles/csvs/human/"
  }
  local1 <- paste(locale, borh, "_", game, "_p", player, ".csv")

  drop.cols1 <- grep("Name$", colnames(sample1), invert = TRUE)
  sample1 <- sample1[grep("Name$|Message$", colnames(sample1), invert = TRUE)]
  colnames(sample1) <- substring(names(sample1), 18)
  cols <- c("Frame", "UnitTags","Pos.X","Pos.Y")
  sample1 <- sample1[cols]
  sample1 <- na.neg(sample1)
  sample1$UnitTags <- lapply(sample1$UnitTags, length)
  sample1 <- mutate_all(sample1, function(x) as.numeric(as.character(x)))
  fwrite(as.data.frame(sample1), file =local1)
  player <- player + 1
  }
  game <<- game + 1
}


filesh <- list.files("gamefiles/JSON/human", pattern="*.JSON", full.names=TRUE)
filesb <- list.files("gamefiles/JSON/bot", pattern="*.JSON", full.names=TRUE)
lapply(filesh, humanCSV)
lapply(filesb, botCSV)

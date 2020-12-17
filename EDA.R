library("usethis")
data = read.csv('dataset_Facebook.csv', head = TRUE, sep=";")
data[is.na(data)] = 0
data$Type[data$Type == 'Photo'] <- 0
data$Type[data$Type == 'Status'] <- 1
data$Type[data$Type == 'Link'] <- 2
data$Type[data$Type == 'Video'] <- 3



set.seed(1)
nrows = dim(data)[1]
data = sapply(data, as.numeric)
shuffled.indexes = sample.int(nrows)
data = as.data.frame(data)
capOutlier <- function(x){
  qnt <- quantile(x, probs=c(.25, .75), na.rm = T)
  caps <- quantile(x, probs=c(.05, .80), na.rm = T)
  H <- 1.5 * IQR(x, na.rm = T)
  x[x < (qnt[1] - H)] <- caps[1]
  x[x > (qnt[2] + H)] <- caps[2]
  return(x)
}
for(col in 1:ncol(data)){
  if(col != 2){
    data[, col]=capOutlier(data[, col])
  }
}

##make 90-10 slpit.

size.train = floor(nrows*0.9)
ind.train = shuffled.indexes[1:size.train]
ind.valid = shuffled.indexes[(size.train + 1):nrows]
data.train = data[ind.train, ]
data.valid = data[ind.valid, ]

data.train = data.train[,c(1,2,3,4,5,6,7,19,9,10,11,12,13,14,15,16,17,18,8)]
data.valid = data.valid[,c(1,2,3,4,5,6,7,19,9,10,11,12,13,14,15,16,17,18,8)]

data.train = transform(data.train, Type = as.integer(Type))
data.valid = transform(data.valid, Type = as.integer(Type))

response.train = data.train[, 19]
response.valid = data.valid[, 19]


library(nnet) # Fits neural net models
set.seed(74364139)

data.train = data.train.raw
data.train = data.train[, -14]
data.valid = data.valid.raw
data.valid = data.valid[, -14]
Y.train = Y.train.raw
Y.valid = Y.valid.raw


rescale <- function(x1,x2){
  for(col in 1:ncol(x1)){
    a <- min(x2[,col])
    b <- max(x2[,col])
    x1[,col] <- (x1[,col]-a)/(b-a)
  }
  x1
}

X.train = rescale(data.train, data.train)
X.valid = rescale(data.valid, data.valid) # Be careful with the order


### Before we can fit the model, we need to choose tuning parameters. For
### now, let's just use 2 hidden nodes and shrinkage of 0.01.
n.hidden = 2
shrink = 0.01
fit.nnet = nnet(y = Y.train, x = X.train, linout = TRUE, size = n.hidden,
                decay = shrink, maxit = 500)


### so we can go back and recover the winner.
n.nnets = 20 # Number of times to re-fit

### Container for SSEs
all.SSEs = rep(0, times = 20)

### Container for models.
### Note: I have used a list here, because this is the only data type in R
###       that can contain complicated objects.
all.nnets = list(1:20)

for(i in 1:n.nnets){
  ### Fit model. We can set the input "trace" to FALSE to suppress the
  ### printed output from the nnet() function.
  this.nnet = nnet(y = Y.train, x = X.train, linout = TRUE, size = n.hidden,
                   decay = shrink, maxit = 500, trace = FALSE)
  
  ### Get the model's SSE
  this.SSE = this.nnet$value
  
  ### Store results. We have to use double square brackets when storing or
  ### retrieving from a list.
  all.SSEs[i] = this.SSE
  all.nnets[[i]] = this.nnet
}

### Get the best model. We have to use double square brackets when storing or
### retrieving from a list.
ind.best = which.min(all.SSEs)
fit.nnet.best = all.nnets[[ind.best]]


### Get predictions and MSPE
pred.nnet = predict(fit.nnet.best, X.valid)
MSPE.nnet = get.MSPE(Y.valid, pred.nnet)



###########################################################################
### So far, so good, but we haven't done any tuning yet. Let's use      ###
### 10-fold CV to choose between 1/5/9 hidden nodes, and 0/1/5/10       ###
### shrinkage. To avoid bad local minima, re-fit each model 20 times    ###
###########################################################################

M = 20 # Number of times to re-fit each model
set.seed(1)
### Define parameter values and use expand.grid() to get all combinations
all.n.hidden = c(1, 3,  5, 7, 9)
all.shrink = c(0.01, 0.1, 0.5, 0, 1, 5, 10)
all.pars = expand.grid(n.hidden = all.n.hidden,
                       shrink = all.shrink)
n.pars = nrow(all.pars) # Number of parameter combinations

K = 5 # Number of folds

### Create folds
folds = get.folds(n, K)

### Create container for MSPEs
CV.MSPEs = array(0, dim = c(K, n.pars))


for(i in 1:K){
  ### Print progress update
  print(paste0(i, " of ", K))
  
  ### Split data and rescale predictors
  data.train = data[folds != i,]
  X.train.raw = data.train[,-5]
  X.train = rescale(X.train.raw, X.train.raw)
  Y.train = data.train[,5]
  
  data.valid = data[folds == i,]
  X.valid.raw = data.valid[,-5]
  X.valid = rescale(X.valid.raw, X.train.raw)
  Y.valid = data.valid[,5]
  
  
  ### Fit neural net models for each parameter combination. A second 
  ### for loop will make our life easier here
  for(j in 1:n.pars){
    ### Get current parameter values
    this.n.hidden = all.pars[j,1]
    this.shrink = all.pars[j,2]
    
    ### We need to run nnet multiple times to avoid bad local minima. Create
    ### containers to store the models and their errors.
    all.nnets = list(1:M)
    all.SSEs = rep(0, times = M)
    
    ### We need to fit each model multiple times. This calls for another
    ### for loop.
    for(l in 1:M){
      ### Fit model
      fit.nnet = nnet(X.train, Y.train, linout = TRUE, size = this.n.hidden,
                      decay = this.shrink, maxit = 500, trace = FALSE)
      
      ### Get model SSE
      SSE.nnet = fit.nnet$value
      
      ### Store model and its SSE
      all.nnets[[l]] = fit.nnet
      all.SSEs[l] = SSE.nnet
    }
    
    ### Get best fit using current parameter values
    ind.best = which.min(all.SSEs)
    fit.nnet.best = all.nnets[[ind.best]]
    
    ### Get predictions and MSPE, then store MSPE
    pred.nnet = predict(fit.nnet.best, X.valid)
    MSPE.nnet = get.MSPE(Y.valid, pred.nnet)
    
    CV.MSPEs[i, j] = MSPE.nnet # Be careful with indices for CV.MSPEs
  }
}


### We can now make an MSPE boxplot. It would be nice to have more 
### informative names though. We can construct names from all.pars
### using the paste0() function.
names.pars = paste0(all.pars$n.hidden,",",
                    all.pars$shrink)
colnames(CV.MSPEs) = names.pars

### Make boxplot
boxplot(CV.MSPEs, las = 2, main = "MSPE Boxplot")


### Get relative MSPEs and make boxplot
CV.RMSPEs = apply(CV.MSPEs, 1, function(W) W/min(W))
CV.RMSPEs = t(CV.RMSPEs)
boxplot(CV.RMSPEs, las = 2, main = "RMSPE Boxplot")


### so we can go back and recover the winner.
n.nnets1 = 20 # Number of times to re-fit

### Container for SSEs
all.SSEs1 = rep(0, times = 20)

### Container for models.
### Note: I have used a list here, because this is the only data type in R
###       that can contain complicated objects.
all.nnets1 = list(1:20)

for(i in 1:n.nnets1){
  ### Fit model. We can set the input "trace" to FALSE to suppress the
  ### printed output from the nnet() function.
  this.nnet.tuned = nnet(y = Y.train, x = X.train, linout = TRUE, size = 9,
                   decay = 0.001, maxit = 500, trace = FALSE)
  
  ### Get the model's SSE
  this.SSE1 = this.nnet.tuned$value
  
  ### Store results. We have to use double square brackets when storing or
  ### retrieving from a list.
  all.SSEs1[i] = this.SSE1
  all.nnets1[[i]] = this.nnet.tuned
}

### Get the best model. We have to use double square brackets when storing or
### retrieving from a list.
ind.best1 = which.min(all.SSEs1)
fit.nnet.best1 = all.nnets1[[ind.best1]]


### Get predictions and MSPE
pred.nnet1 = predict(fit.nnet.best1, X.valid)
MSPE.nnet1 = get.MSPE(Y.valid, pred.nnet1)
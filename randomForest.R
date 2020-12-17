data.train = data.train.raw
data.valid = data.valid.raw
Y.train = Y.train.raw
Y.valid = Y.valid.raw

library(randomForest)

set.seed(36597804)


fit.rf.1 = randomForest(medv ~ ., data = data.train, importance = T)

### Plotting this RF object lets us see if we used enough trees. If not, you can
### change this by re-running randomForest() and setting ntree to a higher
### number (default is 500)
plot(fit.rf.1)

### We can get variable importance measures using the importance() function, and
### we can plot them using VarImpPlot()
importance(fit.rf.1)
varImpPlot(fit.rf.1)

### We can get out-of-bag (OOB) error directly from the predict() function. 
### Specifically, if we don't include a new dataset, R gives the OOB predictions
### on the training set.
OOB.pred.1 = predict(fit.rf.1)
(OOB.MSPE.1 = get.MSPE(Y.train, OOB.pred.1))

###############################################################################
### Now, let's look at how to tune random forests using OOB error. We will  ###
### consider mtry = 1,2,3,4 and nodesize = 2, 5, 8.                         ###
###############################################################################

### Set parameter values
all.mtry = 1:14
all.nodesize = c(2, 5, 8, 11)
all.pars = expand.grid(mtry = all.mtry, nodesize = all.nodesize)
n.pars = nrow(all.pars)

### Number of times to replicate process. OOB errors are based on bootstrapping,
### so they are random and we should repeat multiple runs
M = 5

### Create container for OOB MSPEs
OOB.MSPEs = array(0, dim = c(M, n.pars))

for(i in 1:n.pars){
  ### Print progress update
  print(paste0(i, " of ", n.pars))
  
  ### Get current parameter values
  this.mtry = all.pars[i,"mtry"]
  this.nodesize = all.pars[i,"nodesize"]
  
  ### Fit random forest models for each parameter combination
  ### A second for loop will make our life easier here
  for(j in 1:M){
    ### Fit model using current parameter values. We don't need variable
    ### importance measures here and getting them takes time, so set
    ### importance to F
    fit.rf = randomForest(medv ~ ., data = data.train, importance = F,
                          mtry = this.mtry, nodesize = this.nodesize)
    
    ### Get OOB predictions and MSPE, then store MSPE
    OOB.pred = predict(fit.rf)
    OOB.MSPE = get.MSPE(Y.train, OOB.pred)
    
    OOB.MSPEs[j, i] = OOB.MSPE # Be careful with indices for OOB.MSPEs
  }
}


### We can now make an MSPE boxplot. First, add column names to indicate
### which parameter combination was used. Format is mtry-nodesize
names.pars = paste0(all.pars$mtry,"-",
                    all.pars$nodesize)
colnames(OOB.MSPEs) = names.pars

### Make boxplot
boxplot(OOB.MSPEs, las = 2, main = "MSPE Boxplot")


### Get relative MSPEs and make boxplot
OOB.RMSPEs = apply(OOB.MSPEs, 1, function(W) W/min(W))
OOB.RMSPEs = t(OOB.RMSPEs)
boxplot(OOB.RMSPEs, las = 2, main = "RMSPE Boxplot")

### Zoom in on the competitive models
boxplot(OOB.RMSPEs, las = 2, main = "RMSPE Boxplot", ylim = c(1, 1.02))



### Based on the RMSPE boxplot, the model with mtry=4 and nodesize=2 looks best
### to me. Let's fit this model and see how it compares to the default one from
### above.
fit.rf.2 = randomForest(medv ~ ., data = data.train, importance = T,
                        mtry = 6, nodesize = 2)

### Did we use enough trees?
plot(fit.rf.2)

### How important are the predictors?
varImpPlot(fit.rf.2)

### What is the OOB error?
OOB.pred.2 = predict(fit.rf.2)
(OOB.MSPE.2 = get.MSPE(Y.train, OOB.pred.2))

### How about the SMSE
sample.pred.2 = predict(fit.rf.2, data)
(SMSE.2 = get.MSPE(data$alcohol, sample.pred.2))

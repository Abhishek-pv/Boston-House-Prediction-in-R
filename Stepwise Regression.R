data.train = data.train.raw
data.valid = data.valid.raw
Y.train = Y.train.raw
Y.valid = Y.valid.raw

library(dplyr)
library(leaps) 


shuffle = function(X){
  new.order = sample.int(length(X))
  new.X = X[new.order]
  return(new.X)
}

get.MSPE = function(Y, Y.hat){
  return(mean((Y - Y.hat)^2))
}

predict.matrix = function(fit.lm, X.mat){
  coeffs = fit.lm$coefficients
  Y.hat = X.mat %*% coeffs
  return(Y.hat)
}


### Create a container to store MSPEs
all.models = c("stepwise.AIC", "stepwise.BIC")
all.MSPEs = rep(0, times = length(all.models))
names(all.MSPEs) = all.models

##########################################
### Stepwise selection via AIC and BIC ###
##########################################

fit.start = lm(medv ~ 1, data = data.train)
fit.end = lm(medv ~ .^2, data = data.train)

step.AIC = step(fit.start, list(upper = fit.end), k = 2)
step.BIC = step(fit.start, list(upper = fit.end), k = log(nrow(data.train)),
                trace = 0)

pred.step.AIC = predict(step.AIC, data.valid)
pred.step.BIC = predict(step.BIC, data.valid)

err.step.AIC = get.MSPE(Y.valid, pred.step.AIC)
err.step.BIC = get.MSPE(Y.valid, pred.step.BIC)

all.MSPEs["stepwise.AIC"] = err.step.AIC
all.MSPEs["stepwise.BIC"] = err.step.BIC

### First we need to set the number of folds
K = 10

### Construct folds
### Don't attach fold labels to dataset because we would just have
### to remove this later
n = nrow(data)
n.fold = n/K # Approximate number of observations per fold
n.fold = ceiling(n.fold)
ordered.ids = rep(1:10, each = n.fold)
ordered.ids = ordered.ids[1:n]
fold.ids = shuffle(ordered.ids)
n.models = c("stepwise.AIC", "stepwise.BIC") # Number of candidate models
m = array(0, dim = c(K,length(n.models)))
colnames(m) = n.models

n.models1 = c("stepwise.AIC", "stepwise.BIC") # Number of candidate models
v = array(0, dim = c(K,length(n.models1)))
colnames(v) = n.models1

n.models2 = c("stepwise.AIC", "stepwise.BIC") # Number of candidate models
mo = array(0, dim = c(K,length(n.models2)))
colnames(mo) = n.models2

CV.models = c("stepwise.AIC", "stepwise.BIC")
errs.CV = array(0, dim = c(K,length(CV.models)))
colnames(errs.CV) = CV.models
for(i in 1:K){
  print(paste0(i, " of ", K))
  
  ### Construct training and validation sets by either removing
  ### or extracting the current fold. 
  ### Also, get the response vectors
  data.train = data[fold.ids != i,]
  data.valid = data[fold.ids == i,]
  Y.train = data.train$medv
  Y.valid = data.train$medv
  
  ##########################################
  ### Stepwise selection via AIC and BIC ###
  ##########################################
  
  fit.start = lm(medv ~ 1, data = data.train)
  fit.end = lm(medv ~ .^2, data = data.train)
  
  ### These functions will run several times each. We don't need
  ### to print out all the details, so set trace = 0.
  step.AIC = step(fit.start, list(upper = fit.end), k=2,
                  trace = 0)
  step.BIC = step(fit.start, list(upper = fit.end), k = log(nrow(data.train)),
                  trace = 0)
  print(step.BIC)
  
  pred.step.AIC = predict(step.AIC, data.valid)
  pred.step.BIC = predict(step.BIC, data.valid)
  
  err.step.AIC = get.MSPE(Y.valid, pred.step.AIC)
  err.step.BIC = get.MSPE(Y.valid, pred.step.BIC)
  
  ### Store errors in errs.CV, which has two dimensions, so 
  ### we need two indices
  errs.CV[i, "stepwise.AIC"] = err.step.AIC
  errs.CV[i, "stepwise.BIC"] = err.step.BIC
  
  this.AIC = extractAIC(step.AIC)[2]
  m[i, "stepwise.AIC"] = this.AIC
  this.BIC = extractAIC(step.BIC, k = log(nrow(data.train)))[2]
  m[i, "stepwise.BIC"] = this.BIC
  
  this.AIC = extractAIC(step.AIC)[1]
  v[i, "stepwise.AIC"] = this.AIC
  this.BIC = extractAIC(step.BIC, k = log(nrow(data.train)))[1]
  v[i, "stepwise.BIC"] = this.BIC
}

### Get the optimal model for AIC and BIC
AIC.ind = which.min(m[, 'stepwise.AIC'])
BIC.ind = which.min(m[, 'stepwise.BIC'])
AIC.model = step.BIC[AIC.ind]
BIC.ind = which.min(all.BICs)
BIC.model = all.subsets.models[BIC.ind,]

### Now that we have multiple estimates of the models' MSPEs, let's
### make a boxplot
boxplot(errs.CV, main = "CV Error Estimates")


### Finally, we can get the relative MSPEs and make the corresponding
### boxplot. See Lecture 3 for details of the apply() function.
rel.errs.CV = apply(errs.CV, 1, function(W){
  best = min(W)
  return(W / best)
})
rel.errs.CV = t(rel.errs.CV) # Re-orient output

boxplot(rel.errs.CV, main = "Relative CV Error Estimates")

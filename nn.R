X.train.best = select(X.train, "crim", "tax", "rm", "age", "dis", "rad", "ptratio", "black", "lstat")
X.valid.best = select(X.valid, "crim", "tax", "rm", "age", "dis", "rad", "ptratio", "black", "lstat")

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
  this.nnet.tuned = nnet(y = Y.train, x = X.train.best, linout = TRUE, size = 9,
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
pred.nnet.best = predict(fit.nnet.best1, X.valid)
MSPE.nnet.best = get.MSPE(Y.valid, pred.nnet.best)
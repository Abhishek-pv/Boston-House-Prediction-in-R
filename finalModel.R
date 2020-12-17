set.seed(420)
data.train = data.train.raw
data.valid = data.valid.raw
Y.train = Y.train.raw
Y.valid = Y.valid.raw

K = 10
n= nrow(data.train)
folds = get.folds(n, K)
all.models = c("LS", "Poly", "Ridge", "LASSO-Min", "LASSO-1se", "GAM", "NNET","tree", "tree.min", "tree.1se", "rf", "boosting")
all.MSPEs = array(0, dim = c(K, length(all.models)))
colnames(all.MSPEs) = all.models
#Nested Cross Validation

for(i in 1:K){
  ### Print progress update
  print(paste0(i, " of ", K))
  
  ### Split data and rescale predictors
  data.train = data[folds != i,]
  X.train = data.train[,-14]
  X.train.nn = rescale(X.train, X.train)
  Y.train = data.train[,14]
  
  data.valid = data[folds == i,]
  X.valid = data.valid[,-14]
  X.valid.nn = rescale(X.valid, X.train)
  Y.valid = data.valid[,14]
  
  #####################################
  #############  LM ##################
  #####################################
  fit.lm = lm(medv ~ ., data = data.train)
  pred.lm = predict(fit.lm, data.valid)
  all.MSPEs[i,"LS"] = get.MSPE(Y.valid, pred.lm)
  
  #####################################
  #############  POLY ##################
  #####################################
  fit.poly = lm(medv ~ (.)^2, data = data.train)
  pred.poly = predict(fit.poly, data.valid)
  all.MSPEs[i,"Poly"] = get.MSPE(Y.valid, pred.poly)
  
  #####################################
  #############  GAMS ##################
  #####################################
  fit.gam = gam(medv ~ s(crim)+s(nox)+s(rm)+s(age)+s(dis)+s(rad, k=5)+s(ptratio)+s(black)+s(lstat)+lstat:tax, data=data.train)
  pred.gam = predict(fit.gam, data.valid)
  all.MSPEs[i,"GAM"] = get.MSPE(Y.valid, pred.gam)
  
  #######################################################################
  ### Now we can do the LASSO. This model is fit using the glmnet() ###
  #######################################################################
  matrix.train.raw = model.matrix(medv ~ ., data = data.train)
  matrix.train = matrix.train.raw[,-1]
  all.LASSOs = cv.glmnet(x = matrix.train, y = Y.train)
  
  ### Get both 'best' lambda values using $lambda.min and $lambda.1se
  lambda.min = all.LASSOs$lambda.min
  lambda.1se = all.LASSOs$lambda.1se
  
  ### Get the coefficients for our two 'best' LASSO models
  coef.LASSO.min = predict(all.LASSOs, s = lambda.min, type = "coef")
  coef.LASSO.1se = predict(all.LASSOs, s = lambda.1se, type = "coef")
  
  ### Get which predictors are included in our models (i.e. which 
  ### predictors have non-zero coefficients)
  included.LASSO.min = predict(all.LASSOs, s = lambda.min, 
                               type = "nonzero")
  print(included.LASSO.min)
  included.LASSO.1se = predict(all.LASSOs, s = lambda.1se, 
                               type = "nonzero")
  
  matrix.valid.LASSO.raw = model.matrix(medv ~ ., data = data.valid)
  matrix.valid.LASSO = matrix.valid.LASSO.raw[,-1]
  pred.LASSO.min = predict(all.LASSOs, newx = matrix.valid.LASSO,
                           s = lambda.min, type = "response")
  pred.LASSO.1se = predict(all.LASSOs, newx = matrix.valid.LASSO,
                           s = lambda.1se, type = "response")
  
  ### Calculate MSPEs and store them
  MSPE.LASSO.min = get.MSPE(Y.valid, pred.LASSO.min)
  all.MSPEs[i, "LASSO-Min"] = MSPE.LASSO.min
  
  MSPE.LASSO.1se = get.MSPE(Y.valid, pred.LASSO.1se)
  all.MSPEs[i, "LASSO-1se"] = MSPE.LASSO.1se
  
  #######################################################################
  ### Next, let's do ridge regression. This model is fit using the    ###                                   ###
  #######################################################################
  lambda.vals = seq(from = 0, to = 100, by = 0.05)
  fit.ridge = lm.ridge(medv ~ ., lambda = lambda.vals, 
                       data = data.train)
  ind.min.GCV = which.min(fit.ridge$GCV)
  lambda.min = lambda.vals[ind.min.GCV]
  
  ### Get coefficients corresponding to best lambda value
  ### We can get the coefficients for every value of lambda using
  ### the coef() function on a ridge regression object
  all.coefs.ridge = coef(fit.ridge)
  coef.min = all.coefs.ridge[ind.min.GCV,]
  
  ### We will multiply the dataset by this coefficients vector, but 
  ### we need to add a column to our dataset for the intercept and 
  ### create indicators for our categorical predictors. A simple
  ### way to do this is using the model.matrix() function from last
  ### week.
  matrix.valid.ridge = model.matrix(medv ~ ., data = data.valid)
  pred.ridge = matrix.valid.ridge %*% coef.min
  
  ### Now we just need to calculate the MSPE and store it
  MSPE.ridge = get.MSPE(Y.valid, pred.ridge)
  all.MSPEs[i, "Ridge"] = MSPE.ridge
  
  ##NNET
  n.nnets1 = 20
  all.nnets1 = list(1:20)
  all.SSEs1 = rep(0, times = 20)
  
  for(j in 1:n.nnets1){
    ### Fit model. We can set the input "trace" to FALSE to suppress the
    ### printed output from the nnet() function.
    this.nnet.tuned = nnet(y = Y.train, x = X.train.nn, linout = TRUE, size = 9,
                           decay = 0.001, maxit = 500, trace = FALSE)
    
    ### Get the model's SSE
    this.SSE1 = this.nnet.tuned$value
    
    ### Store results. We have to use double square brackets when storing or
    ### retrieving from a list.
    all.SSEs1[j] = this.SSE1
    all.nnets1[[j]] = this.nnet.tuned
  }
  
  ### Get the best model. We have to use double square brackets when storing or
  ### retrieving from a list.
  ind.best1 = which.min(all.SSEs1)
  fit.nnet.best1 = all.nnets1[[ind.best1]]
  
  
  ### Get predictions and MSPE
  pred.nnet1 = predict(fit.nnet.best1, X.valid.nn)
  MSPE.nnet1 = get.MSPE(Y.valid, pred.nnet1)
  
  all.MSPEs[i, "NNET"] = MSPE.nnet1
  
  ##trees
  fit.tree = rpart(medv ~ ., data = data.train, cp = 0)
  info.tree = fit.tree$cptable
  ind.min = which.min(info.tree[,"xerror"])
  CP.min.raw = info.tree[ind.min, "CP"]
  
  if(ind.min == 1){
    CP.min = CP.min.raw
  } else{
    CP.above = info.tree[ind.min-1, "CP"]
    CP.min = sqrt(CP.min.raw * CP.above)
  }
  
  fit.tree.min = prune(fit.tree, cp = CP.min)
  err.min = info.tree[ind.min, "xerror"]
  se.min = info.tree[ind.min, "xstd"]
  threshold = err.min + se.min
  
  ind.1se = min(which(info.tree[1:ind.min,"xerror"] < threshold))
  CP.1se.raw = info.tree[ind.1se, "CP"]
  if(ind.1se == 1){
    CP.1se = CP.1se.raw
  } else{
    CP.above = info.tree[ind.1se-1, "CP"]
    CP.1se = sqrt(CP.1se.raw * CP.above)
  }
  fit.tree.1se = prune(fit.tree, cp = CP.1se)
  
  predict.tree = predict(fit.tree, data.valid)
  predict.tree.min = predict(fit.tree.min, data.valid)
  predict.tree.1se = predict(fit.tree.1se, data.valid)
  
  
  MSPE.tree = get.MSPE(Y.valid, predict.tree)
  MSPE.tree.1se = get.MSPE(Y.valid, predict.tree.1se)
  MSPE.tree.min = get.MSPE(Y.valid, predict.tree.min)
  
  all.MSPEs[i, "tree"] = MSPE.tree
  all.MSPEs[i, "tree.min"] = MSPE.tree.min
  all.MSPEs[i, "tree.1se"] = MSPE.tree.1se
  
  ##rf
  fit.rf.2 = randomForest(medv ~ ., data = data.train, importance = T,
                          mtry = 6, nodesize = 2)
  sample.pred.2 = predict(fit.rf.2, data.valid)
  SMSE.2 = get.MSPE(Y.valid, sample.pred.2)
  all.MSPEs[i, "rf"] = SMSE.2
  
  ##boosting
  
  fit.gbm.best = gbm(medv ~ ., data = data.train, distribution = "gaussian", 
                     n.trees = 10000, interaction.depth = 5, shrinkage = 0.1, bag.fraction = 0.8)
  
  n.trees.best = gbm.perf(fit.gbm.best, plot.it = F) * 2 # Number of trees
  
  pred.best = predict(fit.gbm.best, data.valid, n.trees.best)
  MSPE.best = get.MSPE(Y.valid, pred.best)
  
  all.MSPEs[i, "boosting"] = MSPE.best
  
}



boxplot(all.MSPEs,
        main = "Boxplot of MSPE for 10 fold Nested CV for the best models")

rel.all.MSPEs = apply(all.MSPEs, 1, function(W){
  best = min(W)
  return(W / best)
})
rel.all.MSPEs = t(rel.all.MSPEs)

boxplot(rel.all.MSPEs,xlab= "models",las=2,
        main = "Boxplot of RMSPE for 10 fold CV for the best models")




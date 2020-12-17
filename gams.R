data.train = data.train.raw
data.valid = data.valid.raw
Y.train = Y.train.raw
Y.valid = Y.valid.raw
library(mgcv)
### This script contains randomness, so we need to set the seed
set.seed(60597418)
fit.gam = gam(medv ~ s(crim)+s(nox)+s(rm)+s(age)+s(dis)+s(rad, k=5)+s(ptratio)+s(black)+s(lstat)+lstat:tax, data=data.train)
pred.gam = predict(fit.gam, data.valid)
MSPE.gam = get.MSPE(Y.valid, pred.gam) # Our helper function
print(MSPE.gam)

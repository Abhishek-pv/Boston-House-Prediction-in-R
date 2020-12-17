data.train = data.train.raw
data.valid = data.valid.raw
Y.train = Y.train.raw
Y.valid = Y.valid.raw
fit.best = lm(medv ~ lstat + ptratio + rm + rm:tax + lstat:tax + ptratio:tax + lstat:rm, data = data.train)
predict.best = predict(fit.best, data.valid)
MSPE.best = get.MSPE(Y.valid, predict.best)

fit.best.2 = lm(formula = medv ~ lstat + rm + ptratio + tax + rad + crim + 
                  lstat:rm + lstat:tax + rm:tax, data = data.train)
predict.best.2 = predict(fit.best.2, data.valid)
MSPE.best.2 = get.MSPE(Y.valid, predict.best.2)
print(MSPE.best.2)

fit.all.best = lm(medv ~ crim+nox+rm+age+dis+rad+ptratio+black+lstat+lstat:tax, data=data.train)
predict.best.all = predict(fit.all.best, data.valid)
MSPE.best.all = get.MSPE(Y.valid, predict.best.all)

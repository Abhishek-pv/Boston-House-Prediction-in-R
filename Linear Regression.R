data.train = data.train.raw
data.valid = data.valid.raw
Y.train = Y.train.raw
Y.valid = Y.valid.raw

fit.all = lm(medv ~ . , data = data.train)
fit.all.2 = lm(medv ~ (.)^2, data = data.train)

predict.all = predict(fit.all, data.valid)
predict.all.2 = predict(fit.all.2, data.valid)

get.MSPE = function(Y, Y.hat) {
  residuals = Y - Y.hat
  resid.sq = residuals ^ 2
  SSPE = sum(resid.sq)
  MSPE = SSPE / length(Y)
  return(MSPE)
}

MSPE.all = get.MSPE(Y.valid, predict.all)
MSPE.all.2 = get.MSPE(Y.valid, predict.all.2)

print(MSPE.all)
print(MSPE.all.2)
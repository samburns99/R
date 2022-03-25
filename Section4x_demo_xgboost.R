# --------------------------------------------------------
# ST4061 / ST6041
# 2021-2022
# Eric Wolsztynski
# ...
# Section 4: demo code for xgboost (Extreme GB)
# --------------------------------------------------------

rm(list=ls())   # clear the environment
library(ISLR)   # contains the data
library(xgboost) # XGBoost... and xgb.DMatrix()
library(caret)   
set.seed(4061)  # for reproducibility

# Set up the data (take a subset of the Hitters dataset)
data(Hitters)
Hitters = na.omit(Hitters)
dat = Hitters
n = nrow(dat)
NC = ncol(dat)
# Change the response variable to a factor to make this a 
# classification problem:
dat$Salary = as.factor(ifelse(Hitters$Salary>median(Hitters$Salary),
						"High","Low"))

# Data partition
itrain = sample(1:n, size=round(.7*n))
dat.train = dat[itrain,]
dat.validation = dat[-itrain,] # independent validation set for later
# x = select(dat.train,-"Salary") ### if using dplyr
# training set:
x = dat.train
x$Salary = NULL
y = dat.train$Salary
# test set:
x.test = dat.validation
x.test$Salary = NULL
y.test = dat.validation$Salary

# XGBoost... 

set.seed(4061)

# (a) line up the data in the required format
# train set:
xm = model.matrix(y~., data=x)[,-1]
x.xgb = xgb.DMatrix(xm) 
# test set:
xm.test = model.matrix(y.test~., x.test)[,-1] 
x.xgb.test = xgb.DMatrix(xm.test)

# (b) training...
# NB: one can run xgbboost() with default parameters by using:
xgb.ctrl = trainControl(method="none")
# otherwise:
xgb.ctrl = trainControl(method="cv", number=10, returnData=FALSE)
xgb.model = train(x.xgb, y, trControl=xgb.ctrl, method="xgbTree")
# NB: use argument tuneGrid to specify custom grids of values for 
# tuning parameters. Otherwise train() picks its own grids.
xgb.model$bestTune

# (c) testing...
xgb.pred = predict(xgb.model, newdata=x.xgb.test)
confusionMatrix(data=xgb.pred, reference=y.test)

# --------------------------------------------------------
# The below demo code is only for information. There is no need 
# to spend time looking into it for ST4061/ST6041 tests/exam!
#
# There are a number of parameters to be tuned for XGBoost:
modelLookup('xgbTree') 
# All or a subset of these parameters can be tuned in a sequential 
# manner. For each tuning parameter, we can define a grid of 
# potential values and search for an optimal value within that grid.
#
# Careful! Running this code will take some time...
#
# (1) Max number of trees (just an example!):
tune.grid = expand.grid(nrounds = seq(500, 1000, by=100),
  eta = c(0.025, 0.05, 0.1, 0.3),
  max_depth = c(2, 3, 4, 5, 6),
  gamma = 0,
  colsample_bytree = 1,
  min_child_weight = 1,
  subsample = 1) 
xgb.ctrl = trainControl(method="cv", number=10, returnData=FALSE)
xgb.tune = train(x.xgb, y, trControl=xgb.ctrl, tuneGrid=tune.grid, method="xgbTree")
#
# (2) Max tree depth and min child weight (just an example!):
tune.grid = expand.grid(nrounds = seq(500, 1000, by=100),
  eta = xgb.tune$bestTune$eta,
  max_depth = c(1:xgb.tune$bestTune$max_depth+2),
  gamma = 0,
  colsample_bytree = 1,
  min_child_weight = c(1:3),
  subsample = 1) 
xgb.ctrl = trainControl(method="cv", number=10, returnData=FALSE)
xgb.tune = train(x.xgb, y, trControl=xgb.ctrl, tuneGrid=tune.grid, method="xgbTree")
#
# (3) sampling (just an example!):
tune.grid = expand.grid(nrounds = seq(500, 1000, by=100),
  eta = xgb.tune$bestTune$eta,
  max_depth = xgb.tune$bestTune$max_depth,
  gamma = 0,
  colsample_bytree = seq(0.2,1,by=.2),
  min_child_weight = xgb.tune$bestTune$min_child_weight,
  subsample = seq(.5,1,by=.1)) 
xgb.ctrl = trainControl(method="cv", number=10, returnData=FALSE)
xgb.tune = train(x.xgb, y, trControl=xgb.ctrl, tuneGrid=tune.grid, method="xgbTree")
#
# (4) gamma (just an example!):
tune.grid = expand.grid(nrounds = seq(500, 1000, by=100),
  eta = xgb.tune$bestTune$eta,
  max_depth = xgb.tune$bestTune$max_depth,
  gamma = seq(0,1,by=.1),
  colsample_bytree = xgb.tune$bestTune$colsample_bytree,
  min_child_weight = xgb.tune$bestTune$min_child_weight,
  subsample = xgb.tune$bestTune$subsample) 
xgb.ctrl = trainControl(method="cv", number=10, returnData=FALSE)
xgb.tune = train(x.xgb, y, trControl=xgb.ctrl, tuneGrid=tune.grid, method="xgbTree")
#
# (5) learning rate (just an example!):
tune.grid = expand.grid(nrounds = seq(500, 5000, by=100),
  eta = c(0.01,0.02,0.05,0.075,0.1),
  max_depth = xgb.tune$bestTune$max_depth,
  gamma = xgb.tune$bestTune$gamma,
  colsample_bytree = xgb.tune$bestTune$colsample_bytree,
  min_child_weight = xgb.tune$bestTune$min_child_weight,
  subsample = xgb.tune$bestTune$subsample) 
xgb.ctrl = trainControl(method="cv", number=10, returnData=FALSE)
xgb.tune = train(x.xgb, y, trControl=xgb.ctrl, tuneGrid=tune.grid, method="xgbTree")
#
# Then fit:
xgb.ctrl = trainControl(method="cv", number=10, returnData=FALSE)
xgb.tune = train(x.xgb, y, trControl=xgb.ctrl, tuneGrid=tune.grid, method="xgbTree")
# testing:
xgb.pred = predict(xgb.model, newdata=x.xgb.test)
confusionMatrix(data=xgb.pred, reference=y.test)






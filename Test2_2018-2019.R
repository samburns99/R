### Test 2 - 2018/2019
## Q1
# (a)

# The support vectors for an SVM that classifies a 2 factor response variable.

# (b)

# SVMs with different kernels is the likely cause in difference.
# Left plot likely linear kernel
# Right plot likely radial

## Q2
# (a)

# Feed forward neural network

# (b)

# 2 hidden layers
# Sizes 4 and 2 respectively

## Q3

library(leaps)
library(randomForest)
dat = read.csv(file="C:\\Users\\david\\Desktop\\4th Year DSA\\ST4061\\Past CAs\\Q3_dataset.csv")
X = dat
X$Y <- NULL
Y = dat$Y

# (a)

K = 10
set.seed(1)
N = nrow(X)
P = ncol(X)
folds = cut(1:N, K, labels=FALSE)
bics = matrix(NA, nrow=K, ncol=10)
vars = matrix(NA, nrow=K, ncol= P) # fill in the blank for ncol
for(k in 1:K){
  itrain = which(folds != k)
  fwd.mod = regsubsets(x=X[itrain, ], y=Y[itrain], method="forward", nvmax=10)
  bics[k, ] = summary(fwd.mod)$bic
  vars[k, ] = summary(fwd.mod)$which[which.min(bics[k, ]), -1]
}

# (b)

apply(bics, 1, which.min)

# (c)

vars

# (d)

100 * apply(vars, 2, mean)

# (e)

rf.out = randomForest(Y ~ ., data=X)
rf.out$mtry

# (f)

rf.out$importance
nms = colnames(X)
nms[order(rf.out$importance, decreasing=TRUE)]



P
vars
bics
?which


summary(fwd.mod)$which[which.min(bics[1, ]), -1]
summary(fwd.mod)$which[1, ]
summary(fwd.mod)[1,]










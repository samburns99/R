### Test 2 - 2017/2018

library(mlbench)
data(Sonar) 
library(glmnet)
library(tree)
library(randomForest)
library(pROC)

N = nrow(Sonar)
P = ncol(Sonar)-1
M = 150 # size of training set
set.seed(1)
mdata = Sonar[sample(1:N),]
itrain = sample(1:N,M)
x = as.matrix(mdata[,-ncol(mdata)])
y = mdata$Class

# (1)

# 58

# (2)

fwd.mod = regsubsets(x=x[itrain, ], y=y[itrain], method="forward", nvmax=P)
summary(fwd.mod)$bic
names(summary(fwd.mod))

dev.off()
plot(summary(fwd.mod)$bic, pch=20, t='b', xlab='model size', ylab='BIC')

# (3)

# 12 variables

# (4)

lasso.opt = cv.glmnet(as.matrix(x[itrain,]),y[itrain],alpha=1,family="binomial")
lasso.opt$lambda.min

# (5)
# (a)
lasso = glmnet(as.matrix(x[itrain,]),y[itrain],alpha=1, lambda=lasso.opt$lambda.min, family="binomial")
names(lasso)
lasso$dim

cbind(round(lasso$beta[, 1], 3))
lasso$beta[which(lasso$beta != 0)]

# (b)

sum(lasso$beta != 0)

# (6)

tree.model = tree(y[itrain]~., data=data.frame(x[itrain,]))
names(summary(tree.model))
length(summary(tree.model)$used)

# (7)

rf = randomForest(y[itrain]~., data=data.frame(x[itrain,]))
varImpPlot(rf)

# (8)

tree.pred = predict(tree.model, newdata=as.data.frame(x[-itrain,]), type='vector')
table(tree.pred, y[-itrain])

# (9)

tree.roc = roc(response=y[-itrain], predictor=tree.pred[, 1])
plot(tree.roc)

rf.pred = predict(rf,x[-itrain,], type='prob')
rf.roc = roc(response=y[-itrain], predictor=rf.pred[, 1])
par(new=TRUE)
plot(rf.roc, col=4)
legend('bottomright', legend=c('Tree', 'RF'), col=c(1, 4), pch=15, bty='n')



dev.off()




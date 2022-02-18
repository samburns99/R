### 2018-2019 Test 1
## Q1

library(mlbench)
data(Sonar) 
N = nrow(Sonar)
P = ncol(Sonar)-1
M = 150
set.seed(1)
mdata = Sonar[sample(1:N),]
itrain = sample(1:N,M)
x = mdata[,-ncol(mdata)]
y = mdata$Class
xm = as.matrix(x)
#x.train = 
x
y
itrain
# (1)

# 58 observations

# (2)

library(glmnet)
cv.l = cv.glmnet(xm[itrain, ],y[itrain], family="binomial")
cv.l$lambda.min

# (3)
?cv.glmnet
lo = glmnet(xm[itrain, ],y[itrain],family="binomial",lambda=cv.l$lambda.min)
coef(lo)

# (4)

library(tree)
?tree
tree1 = tree(y ~ ., x, subset=itrain)
summary(tree1)

# (5)

library(randomForest)
rf.out = randomForest(y ~ ., x, subset=itrain)
summary(rf.out)
varImpPlot(rf.out)

# (6)

xtest = x[-itrain, ]
rf.pred = predict(rf.out, xtest, type="class")
tree.pred = predict(tree1, xtest, type="class")

(tb.tree = table(tree.pred, y[-itrain]))
(tb.rf = table(rf.pred, y[-itrain]))

1 - sum(diag(tb.tree))/sum(tb.tree)
1 - sum(diag(tb.rf))/sum(tb.rf)

# (7)

library(pROC)
tree.p = predict(tree1, x[-itrain,], 'vector')[,2]
rf.p = predict(rf.out, x[-itrain,], 'prob')[,2]
auc.tree = roc(y[-itrain],tree.p)$auc
auc.rf = roc(y[-itrain],rf.p)$auc
c(auc.tree,auc.rf)












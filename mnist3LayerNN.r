
# read in the dataset
data <- read.csv('mnist.csv', header=FALSE)

totalExamples <- dim(data)[1] # the total number of examples to use (split between training/test)
trainTestSplit <- 0.8 # what portion of examples to use for training
m <- floor(totalExamples * trainTestSplit) # number of training examples
testSetSize <- totalExamples - m # number of test examples

X_train <- data[1:m, -(1)] # training set
y_train <- data[1:m, 1] # training labels
X_test <- data[(m + 1):totalExamples, -(1)] # test set
y_test <- data[(m + 1):totalExamples, 1] # test labels

n <- dim(X_train)[2] # number of features
alpha <- 0.01 # learning rate
hiddenLayerSize <- 800 # size of hidden layer
K <- 10 # number of classes
Y <- diag(K) # output matrix
iterations <- 100 # number of iterations

theta1 <- matrix(runif(hiddenLayerSize * (n + 1))*2-1, nrow=hiddenLayerSize, ncol=(n + 1))
theta2 <- matrix(runif(K * (hiddenLayerSize + 1))*2-1, nrow=K, ncol=(hiddenLayerSize + 1))

# sigmoid function
sigmoid <- function(x)
{
  sig <- 1.0 / (1.0 + exp(-0.01 * x))
  return (sig)
}

# hypothesis function
hyp <- function(a1, theta1, theta2)
{
  z2 <- theta1 %*% a1
  a2 <- c(1, sigmoid(z2))
  z3 <- theta2 %*% a2
  a3 <- sigmoid(z3)
  
  res <- list(a2, a3)
  return (res)
}

startTime <- proc.time();
for (i in 1 : iterations) # update thetas for _all_ training examples
{
  for (k in 1 : m) # update thetas for _each_ training example
  {
    # forward prop
    a1 <- matrix(c(1, t(X_train[k,])), ncol=1)
    hypResult <- hyp(a1, theta1, theta2)
    a2 <- matrix(unlist(hypResult[1]), ncol=1)
    a3 <- matrix(unlist(hypResult[2]), ncol=1)
    
    # back prop
    delta3 <- (a3 - Y[, y_train[k] + 1]) * (a3 * (1 - a3))
    delta2 <- ((t(theta2) %*% delta3) * (a2 * (1 - a2)))[-(1)]
    
    # update thetas
    theta1 <- theta1 - t((alpha * (a1 %*% t(delta2))))
    theta2 <- theta2 - t((alpha * (a2 %*% t(delta3))))
  }
}
sprintf("Training time: %f minutes.", ((proc.time() - startTime)[3]) / 60)

correctCount <- 0; # count number of correct hypotheses
startTime <- proc.time();
for (j in 1 : testSetSize) # check hypothesis on each test example
{
  a1 <- matrix(c(1, t(X_test[j,])), ncol=1)
  hypResult <- hyp(a1, theta1, theta2)
  a3 <- matrix(unlist(hypResult[2]), ncol=1)
  
  maxIndex <- which.max(a3);
  if(maxIndex == y_test[j] + 1) # increment count if hypothesis is correct
  {
    correctCount = correctCount + 1;
  }
}
sprintf("Testing time: %f seconds.", (proc.time() - startTime)[3])
sprintf("Accuracy: %f%%", (correctCount / testSetSize) * 100)
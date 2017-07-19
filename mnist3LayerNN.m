
# read in the dataset
data = csvread('mnist.csv');

totalExamples = size(data)(1); # the total number of examples to use (split between training/test)
trainTestSplit = 0.8; # what portion of examples to use for training
m = floor(totalExamples * trainTestSplit); # number of training examples
testSetSize = totalExamples - m; # number of test examples

X_train = data(1:m, 2:end); # training set
y_train = data(1:m, 1); # training labels
X_test = data(m + 1:totalExamples, 2:end); # test set
y_test = data(m + 1:totalExamples, 1); # test labels

n = size(X_train)(2); # number of features
alpha = 0.01; # learning rate
hiddenLayerSize = 800; # size of hidden layer
K = 10; # number of classes
Y = eye(K); # output matrix
iterations = 100; # number of iterations

# initialise thetas
theta1 = 2 * rand(hiddenLayerSize, n + 1) - 1;
theta2 = 2 * rand(K, hiddenLayerSize + 1) - 1;

# sigmoid function
function sig = sigmoid(x)
  sig = 1.0 ./ (1.0 + exp(-0.01 * x));
endfunction

# hypothesis function
function [a2, a3] = hyp(a1, theta1, theta2)
  z2 = theta1 * a1;
  a2 = [1; sigmoid(z2)];
  z3 = theta2 * a2;
  a3 = sigmoid(z3);
endfunction

startTime = time();
for i = 1 : iterations # update thetas for _all_ training examples
  for k = 1 : m # update thetas for _each_ training example

    # forward prop
    a1 = [1; X_train(k, :)'];
    [a2, a3] = hyp(a1, theta1, theta2);

    # back prop
    delta3 = (a3 - Y(:, y_train(k) + 1)) .* (a3 .* (1 - a3));
    delta2 = ((theta2' * delta3) .* (a2 .* (1 - a2)))(2:end);

    # update thetas
    theta1 = theta1 - (alpha * (a1 * delta2'))';
    theta2 = theta2 - (alpha * (a2 * delta3'))';
    
  endfor
endfor
printf("Training time: %d minutes.\n", (time() - startTime) / 60);

correctCount = 0; # count number of correct hypotheses
startTime = time();
for i = 1 : testSetSize # check hypothesis on each test example

  [a2, a3] = hyp([1; X_test(i, :)'], theta1, theta2);
  
  [maxValue, maxIndex] = max(a3);
  if(maxIndex == y_test(i) + 1) # increment count if hypothesis is correct
    correctCount = correctCount + 1;
  endif
  
endfor
printf("Testing time: %d seconds.\n", time() - startTime);
printf("Accuracy: %d%%\n", (correctCount / testSetSize) * 100);

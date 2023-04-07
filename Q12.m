

%Read the data
df = readmatrix('data.xlsx');

%Create X_train and Y_train

y = df(:,end);
X = df(:,1:4);
Bias = ones(length(y),1);
X_train = [Bias X];
Y_train = y.';

%I selected to make cross-validation for different number of hidden neurons
%in hidden layers. Also lr and epochs can be good hyperparameters. It can
%be also easily done.
n1 = [5 8];
n2 = [4 7];

epochs = 100;
lr = 0.1;

% Cross validation errors with Leave one out CV.
sol = LOO(X_train, Y_train, epochs, lr, n1, n2);





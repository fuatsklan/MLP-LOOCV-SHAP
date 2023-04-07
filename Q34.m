
df = readmatrix('data.xlsx');

%M = NN(weights, derivatives, activations);

y = df(:,end);
X = df(:,1:4);
Bias = ones(length(y),1);
X_train = [Bias X];
Y_train = y.';


epochs = 100;
lr = 0.1;

%Initilize weights 
weights = {rand(5,5), rand(5,6), rand(6,1)};
derivatives = {zeros(5,5), zeros(5,6), zeros(6,1)};
activations = {zeros(5,1), zeros(5,1), zeros(6,1), zeros(1,1)};

%Create ml model and train.
shapeley_matrix = X_train;
M = NN(weights, derivatives, activations);
M.train(X_train, Y_train, epochs, lr)

for i = 1:size(X_train,1)
    for j=1:size(X_train,2)-1
        shapeley_matrix(i,j) = shap(M, X_train(i,:), j,100,  X_train);
    
    
    end
end

shapeley_matrix = shapeley_matrix(:,1:4);

% Avarage impacts of features by avaraging shapeley values.

a_s = mean(shapeley_matrix(:,1));
b_s = mean(shapeley_matrix(:,2));
c_s = mean(shapeley_matrix(:,3));
d_s = mean(shapeley_matrix(:,4));

feature_importance = [a_s b_s c_s d_s];







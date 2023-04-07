

function y = LOO(X, Y, epochs, lr, n1, n2) % for [5, n,n,1] structure

e = [];
for i=1:length(n1)
    for j=1:length(n2)
    % Declaring weight structer aligned with MLP structure    
    weights = {rand(5,n1(i)), rand(n1(i),n2(j)), rand(n2(j),1)};
    derivatives = {zeros(5,n1(i)), zeros(n1(i),n2(j)), zeros(n2(j),1)};
    activations = {zeros(5,1), zeros(n1(i),1), zeros(n2(j),1), zeros(1,1)};
    
    reg = NN(weights, derivatives, activations);
    
    a = reg.LVOCV(X, Y, epochs, lr);
    % CV error array to compare different structered models
    e = [a e];
    
    end
end

y = e;

end
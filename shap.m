

function y = shap(model, X, feature_index,n, X_train) % Estimating Shapley values with Monte-Carlo Sampling.

shapley = 0;
feature_indexes = 1:4;
feature_indexes(:,feature_index) = [];
for i = 1:n
    
    x = X(:,2:end);     % extracting first column since it is bias.
    
    r = randi([1 101],1,1);        % creating a random number between 1-101.
    z = X_train(r,:);     % z is the random sample from X.
    z = z(:,2:end);
    
    r_1 = randsample(size(x,2)-1,1);      % random number to randomly select number of column indexes.
    x_index = randsample(3,r_1).';        % randomly select column indexes.
    z_index = setdiff(feature_indexes, x_index);        % Other than selected indexes
    
    xj = [];
    x_j = [];
    
    % calculating sample with specific feature and without specific column.
    % Then we will make predictions upon that and the difference will be
    % marginal contribution. After M sample we will avarage those and get
    % the shapeley estimation for one sample and single feature.
    for i= 1:4
        if ismember(i,[x_index feature_index])
            xj = [xj x(i)];
        else
            xj = [xj z(i)];
        end
    end
    
    for i = 1:4
        
        if ismember(i,[z_index feature_index] )
            x_j = [x_j z(i)];
        else
            x_j = [x_j x(i)];
        end
    end
    
    %again add bias term to both
    xj = [1 xj];
    x_j = [1 x_j];
    
    %predict the output by constructed model.
    
    m_cont = model.forward(xj)-model.forward(x_j);
    shapley = shapley + m_cont;
    
        
   
end

y = shapley/n;

end


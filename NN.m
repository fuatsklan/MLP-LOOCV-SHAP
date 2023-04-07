% Creating class for Multi Layer Perceptron

classdef NN < handle
    
    properties
        % declare properties
        weights 
        derivatives
        activations
    end
    
    methods
        %Initielize
        function obj = NN(weights, derivatives, activations)

            obj.weights = weights;
            obj.derivatives = derivatives;
            obj.activations = activations;
            
        end       
        %Nonlinear activation function for hidden layers
        function y = sigmoid(obj, x)
            
            y = 1/(1+exp(-x));
            
        end
        
        
        %Derivative of nonlinear activation functiın for calculating
        %derivatives.
        function y_ = sigmoid_derivative(obj, x)
            
            y_ = x .* (1.-x);
            
        end
        %Mean Square Error metrics.
        function meansquareerror = mse(obj, y_true, y_hat)
            
            meansquareerror = mean((y_true-y_hat)^2);
            
        end
        %Gradient descen algorithm to optimize weights and mse loss.
        function gd = gradientdescent(obj, lr)
            
            for i = 1:length(obj.weights)
                
                w = obj.weights{i};
                d = obj.derivatives{i};
                obj.weights{i} = w + d*lr;
                
                
            end
        end
        %forward propagation
        function activations = forward(obj, inputs)
            
            activations = inputs.';
            obj.activations{1} = activations;
            
            for i = 1:length(obj.weights)-1
                
                
                net_inputs = activations.' * obj.weights{i};
                activations = arrayfun(@obj.sigmoid,net_inputs.');
                obj.activations{i+1} = activations;
                
            end
            
            net_inputs = activations.' * obj.weights{end};
            activations = net_inputs;
            obj.activations{end} = activations;
        end
        %Backward propogation to calculate derivatives and deltas.
        function y = backprop(obj, error)
            
            
            for i = length(obj.derivatives):-1:1
                
                act = obj.activations{i+1};
                
                if i == length(obj.derivatives)
                    delta = error;
                    cur_act = obj.activations{i};
                    obj.derivatives{i} = cur_act * delta;
                    error = delta * obj.weights{i}.';
                else
                    delta = error .* obj.sigmoid_derivative(act).';
                    cur_act = obj.activations{i};
                    obj.derivatives{i} = cur_act * delta;
                    error = delta * obj.weights{i}.';
                    

                end
            end
        end
        
        %Trainning model.
        function y = train(obj, X_train, Y_train, epochs, lr)
            
            for i = 1:epochs
                
                sum_errors = 0;
                for j = 1:size(X_train, 1)
                    target = Y_train(j);
                    output = obj.forward(X_train(j,:));
                    error = target - output;
                    obj.backprop(error);
                    obj.gradientdescent(lr);
                    sum_errors = sum_errors + obj.mse(target, output);
                    
                    
                end
                
                %y = sum_errors/size(X_train, 1);
                %a = sum_errors/size(X_train, 1);
                %fprintf("epoch : %i error = %.10f \n ",i,a)
            end
            y = sum_errors/size(X_train, 1);
        end
        
        % Leave one out cross validation.
        function p = LVOCV(obj, X_train, Y_train, epochs, lr)
            
            n = length(Y_train);
            error = 0;
            
            for i = 1:n
                X = X_train;
                Y = Y_train;
                
                X_test = X_train(i,:);
                X(i,:) = [];
                
                Y_test = Y_train(i);
                Y(i) = [];
                
                error =  error + obj.train(X, Y, epochs, lr);
            end
            
            p = (1/n)*error;
            
            end
            
        
    end
    
 
    
end
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split

class LinearRegression:

    def __init__(self, learning_rate, num_iterations, lam):
        self.learning_rate = learning_rate
        self.num_iterations = num_iterations
        self.lam = lam

    def predict(self, X):

        return np.dot(X, self.W) + self.b
    
    def update_weights(self, X, y):

        y_pred = self.predict(X)
        n = X.shape[0]

        loss = (1 / n) * np.sum((y - y_pred) ** 2) - (self.lam ** 2 * np.sum(self.W) ** 2)

        dW = - (2 / n) * np.dot(X.T, y - y_pred) - 2 * self.lam * np.sum(self.W)
        db = - (2 / n) * np.sum(y - y_pred) - 2 * self.lam * np.sum(self.b)

        self.W = self.W - self.learning_rate * dW
        self.b = self.b - self.learning_rate * db

        return loss

    def fit(self, X, y):
        self.n, self.m = X.shape

        self.W = np.zeros(self.m)
        self.b = 0

        for i in range(self.num_iterations):
            loss = self.update_weights(X, y)
            print(f"Loss: {loss}")


def main() : 
      
    # Importing dataset 
    df = pd.read_csv("salary_data.csv") 
    X = df.iloc[:,:-1].values 
    Y = df.iloc[:,1].values 
      
    # Splitting dataset into train and test set 
    X_train, X_test, Y_train, Y_test = train_test_split(
        X, 
        Y, 
        test_size = 0.2, 
        random_state = 0
    ) 
      
    # Model training 
    model = LinearRegression( num_iterations = 100, learning_rate = 0.01, lam=.2) 
    model.fit(X_train, Y_train) 
      
    # Prediction on test set 
    Y_pred = model.predict(X_test) 
    
    print("Predicted values ", np.round( Y_pred[:3], 2))  
    print("Real values      ", Y_test[:3]) 
    print("Trained W        ", round(model.W[0], 2)) 
    print("Trained b        ", round(model.b, 2)) 
      
    # Visualization on test set  
    plt.scatter(X_test, Y_test, color = 'blue') 
    sorted_indices = np.argsort(X_test[:, 0])
    plt.plot(X_test[sorted_indices], Y_pred[sorted_indices], color='orange')
    plt.title('Salary vs Experience') 
    plt.xlabel('Years of Experience') 
    plt.ylabel('Salary') 
    plt.show() 
     
if __name__ == "__main__" :  
    main()
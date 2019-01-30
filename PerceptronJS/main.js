const np = require('numjs');


class Perceptron {
    /* 
        epochs: Total number of training cycles. 
        bias: it's the y-intercep, add more flexibility to our model.  
        leraning_rate: it's a hyper parameter which control how much we are adjusting the weights
    */

    constructor(epochs = 10, bias = 0, learning_rate = .01) {
        this.bias = bias;
        this.learning_rate = learning_rate;
        this.epochs = epochs;
        this.w = null
    }


    predict(x) {
        // Weighted sum
        // The neuron will active if and only if the result is greater-than 0
        return (np.dot(x, this.w).get(0) + this.bias) > 0 ? 1 : 0;
    }

    train(X_train, y_train) {
        let epoch = 1
        this.w = np.zeros(X_train.shape[1])

        while (epoch <= this.epochs) {

            for (let i = 0; i < y_train.shape[0]; i++) {
                let xis = np.array([X_train.get(i, 0), X_train.get(i, 1)]), yi = y_train.get(i);

                let y = this.predict(xis),
                    error = yi - y;

                this.bias += this.learning_rate * error;
                // console.log(this.w.add(xis.multiply(this.learning_rate * error)));

                this.w = this.w.add(xis.multiply(this.learning_rate * error));

            }

            epoch++;
        }

    }

}

const p = new Perceptron();

// Dataset: AND logic gate inputs 
X_train = np.array([
    [0, 0],
    [1, 0],
    [0, 1],
    [1, 1]
]);

// AND logic gate result
y_train = np.array([0, 0, 0, 1]);


p.train(X_train, y_train);

console.log(p.predict(np.array([1, 1])));


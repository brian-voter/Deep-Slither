package net.chrono7.wormsai;

import org.nd4j.linalg.activations.IActivation;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.indexing.BooleanIndexing;
import org.nd4j.linalg.indexing.conditions.GreaterThanOrEqual;
import org.nd4j.linalg.lossfunctions.ILossFunction;
import org.nd4j.linalg.lossfunctions.LossUtil;
import org.nd4j.linalg.ops.transforms.Transforms;
import org.nd4j.linalg.primitives.Pair;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;


public class HuberLoss implements ILossFunction {

    /* This example illustrates how to implements a custom loss function that can then be applied to training your neural net
       All loss functions have to implement the ILossFunction interface
       The loss function implemented here is:
       L = (y - y_hat)^2 +  |y - y_hat|
        y is the true label, y_hat is the predicted output
     */

    private static Logger logger = LoggerFactory.getLogger(HuberLoss.class);

    /*
    Needs modification depending on your loss function
        scoreArray calculates the loss for a single data point or in other words a batch size of one
        It returns an array the shape and size of the output of the neural net.
        Each element in the array is the loss function applied to the prediction and it's true value
        scoreArray takes in:
        true labels - labels
        the input to the final/output layer of the neural network - preOutput,
        the activation function on the final layer of the neural network - activationFn
        the mask - (if there is a) mask associated with the label
     */
    private INDArray scoreArray(INDArray labels, INDArray preOutput, IActivation activationFn, INDArray mask) {
        // This is the output of the neural network, the y_hat in the notation above
        //To obtain y_hat: pre-output is transformed by the activation function to give the output of the neural network
        INDArray output = activationFn.getActivation(preOutput.dup(), true);
        //The score is calculated as the sum of (y-y_hat)^2 + |y - y_hat|

        INDArray yMinusyHat = Transforms.abs(labels.sub(output));

        BooleanIndexing.applyWhere(yMinusyHat, new GreaterThanOrEqual(1), //condition to use linear
                err -> err.doubleValue() - 0.5, // func = linear
                err -> err.doubleValue() * err.doubleValue() / 2); //alternative func = quadratic

        if (mask != null) {
            LossUtil.applyMask(yMinusyHat, mask);
        }
        return yMinusyHat;
    }

    /*
    Remains the same for all loss functions
    Compute Score computes the average loss function across many datapoints.
    The loss for a single datapoint is summed over all output features.
     */
    @Override
    public double computeScore(INDArray labels, INDArray preOutput, IActivation activationFn, INDArray mask, boolean average) {
        INDArray scoreArr = scoreArray(labels, preOutput, activationFn, mask);

        double score = scoreArr.sumNumber().doubleValue();

        if (average) {
            score /= scoreArr.size(0);
        }

        return score;
    }

    /*
    Remains the same for all loss functions
    Compute Score computes the loss function for many datapoints.
    The loss for a single datapoint is the loss summed over all output features.
    Returns an array that is #of samples x size of the output feature
     */
    @Override
    public INDArray computeScoreArray(INDArray labels, INDArray preOutput, IActivation activationFn, INDArray mask) {
        INDArray scoreArr = scoreArray(labels, preOutput, activationFn, mask);
        return scoreArr.sum(1);
    }

    /*
    Needs modification depending on your loss function
        Compute the gradient wrt to the preout (which is the input to the final layer of the neural net)
        Use the chain rule
        In this case L = (y - yhat)^2 + |y - yhat|
        dL/dyhat = -2*(y-yhat) - sign(y-yhat), sign of y - yhat = +1 if y-yhat>= 0 else -1
        dyhat/dpreout = d(Activation(preout))/dpreout = Activation'(preout)
        dL/dpreout = dL/dyhat * dyhat/dpreout
    */
    @Override
    public INDArray computeGradient(INDArray labels, INDArray preOutput, IActivation activationFn, INDArray mask) {
        INDArray output = activationFn.getActivation(preOutput.dup(), true);

        INDArray yMinusyHat = Transforms.abs(labels.sub(output));

        BooleanIndexing.applyWhere(yMinusyHat, new GreaterThanOrEqual(1), //condition to use linear
                err -> err, // func = linear
                err -> 1); //alternative func = quadratic

        //Everything below remains the same
        if (mask != null && LossUtil.isPerOutputMasking(yMinusyHat, mask)) {
            LossUtil.applyMask(yMinusyHat, mask);
        }

        INDArray gradients = (INDArray)activationFn.backprop(preOutput, yMinusyHat).getFirst();
        if (mask != null) {
            LossUtil.applyMask(gradients, mask);
        }

        return gradients;
    }

    //remains the same for a custom loss function
    @Override
    public Pair<Double, INDArray> computeGradientAndScore(INDArray labels, INDArray preOutput, IActivation activationFn, INDArray mask, boolean average) {
        return new Pair<>(
                computeScore(labels, preOutput, activationFn, mask, average),
                computeGradient(labels, preOutput, activationFn, mask));
    }

    @Override
    public String name() {
        return "CustomLossL1L2";
    }


    @Override
    public String toString() {
        return "CustomLossL1L2()";
    }

    public boolean equals(Object o) {
        if (o == this) return true;
        if (!(o instanceof HuberLoss)) return false;
        final HuberLoss other = (HuberLoss) o;
        if (!other.canEqual((Object) this)) return false;
        return true;
    }

    public int hashCode() {
        int result = 1;
        return result;
    }

    protected boolean canEqual(Object other) {
        return other instanceof HuberLoss;
    }
}
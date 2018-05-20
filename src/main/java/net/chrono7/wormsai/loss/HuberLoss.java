package net.chrono7.wormsai.loss;

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

/*
CREDIT: Based on https://github.com/deeplearning4j/dl4j-examples/blob/master/dl4j-examples/src/main/java/org/deeplearning4j/examples/misc/lossfunctions/CustomLossL1L2.java
 */

public class HuberLoss implements ILossFunction {

    private static Logger logger = LoggerFactory.getLogger(HuberLoss.class);

    private INDArray scoreArray(INDArray labels, INDArray preOutput, IActivation activationFn, INDArray mask) {
        // This is the output of the neural network, the y_hat in the notation above
        //To obtain y_hat: pre-output is transformed by the activation function to give the output of the neural network
        INDArray output = activationFn.getActivation(preOutput.dup(), true);

        INDArray yMinusyHat = Transforms.abs(labels.sub(output));

        BooleanIndexing.applyWhere(yMinusyHat, new GreaterThanOrEqual(1), //condition to use linear
                err -> err.doubleValue() - 0.5, // func = linear
                err -> err.doubleValue() * err.doubleValue() / 2); //alternative func = quadratic

        if (mask != null) {
            LossUtil.applyMask(yMinusyHat, mask);
        }
        return yMinusyHat;
    }

    @Override
    public double computeScore(INDArray labels, INDArray preOutput, IActivation activationFn, INDArray mask, boolean average) {
        INDArray scoreArr = scoreArray(labels, preOutput, activationFn, mask);

        double score = scoreArr.sumNumber().doubleValue();

        if (average) {
            score /= scoreArr.size(0);
        }

        return score;
    }

    @Override
    public INDArray computeScoreArray(INDArray labels, INDArray preOutput, IActivation activationFn, INDArray mask) {
        INDArray scoreArr = scoreArray(labels, preOutput, activationFn, mask);
        return scoreArr.sum(1);
    }

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
        return "HuberLoss";
    }


    @Override
    public String toString() {
        return "HuberLoss()";
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
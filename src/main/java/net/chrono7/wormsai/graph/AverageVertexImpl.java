//
// Source code recreated from a .class file by IntelliJ IDEA
// (powered by Fernflower decompiler)
//

package net.chrono7.wormsai.graph;

import org.deeplearning4j.nn.api.Layer;
import org.deeplearning4j.nn.api.MaskState;
import org.deeplearning4j.nn.gradient.Gradient;
import org.deeplearning4j.nn.graph.ComputationGraph;
import org.deeplearning4j.nn.graph.vertex.BaseGraphVertex;
import org.deeplearning4j.nn.graph.vertex.VertexIndices;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.primitives.Pair;

import java.util.Arrays;

public class AverageVertexImpl extends BaseGraphVertex {
    private int[] forwardShape;

    public AverageVertexImpl(ComputationGraph graph, String name, int vertexIndex) {
        this(graph, name, vertexIndex, (VertexIndices[])null, (VertexIndices[])null);
    }

    public AverageVertexImpl(ComputationGraph graph, String name, int vertexIndex, VertexIndices[] inputVertices, VertexIndices[] outputVertices) {
        super(graph, name, vertexIndex, inputVertices, outputVertices);
    }

    public boolean hasLayer() {
        return false;
    }

    public Layer getLayer() {
        return null;
    }

    public INDArray doForward(boolean training) {
        if (!this.canDoForward()) {
            throw new IllegalStateException("Cannot do forward pass: input not set");
        } else {
            this.forwardShape = Arrays.copyOf(this.inputs[0].shape(), this.inputs[0].rank());
            switch(this.inputs[0].rank()) {
                case 2:
                    return Nd4j.sum(inputs[0], 1).div(inputs[0].columns());
//                case 3:
//                    return this.inputs[0].get(new INDArrayIndex[]{NDArrayIndex.all(), NDArrayIndex.interval(this.from, this.to, true), NDArrayIndex.all()});
//                case 4:
//                    return this.inputs[0].get(new INDArrayIndex[]{NDArrayIndex.all(), NDArrayIndex.interval(this.from, this.to, true), NDArrayIndex.all(), NDArrayIndex.all()});
                default:
                    throw new UnsupportedOperationException("Cannot get subset for activations of rank " + this.inputs[0].rank());
            }
        }
    }

    public Pair<Gradient, INDArray[]> doBackward(boolean tbptt) {
        if (!this.canDoBackward()) {
            throw new IllegalStateException("Cannot do backward pass: error not set");
        } else {
            INDArray out = Nd4j.zeros(this.forwardShape);
            switch(this.forwardShape.length) {
                case 2:
                    for (int i = 0; i < out.columns(); i++) {
                        out.putColumn(i, this.epsilon.dup());
                    }
                    out.div(out.columns());
                    break;
//                case 3:
//                    out.put(new INDArrayIndex[]{NDArrayIndex.all(), NDArrayIndex.interval(this.from, this.to, true), NDArrayIndex.all()}, this.epsilon);
//                    break;
//                case 4:
//                    out.put(new INDArrayIndex[]{NDArrayIndex.all(), NDArrayIndex.interval(this.from, this.to, true), NDArrayIndex.all(), NDArrayIndex.all()}, this.epsilon);
//                    break;
                default:
                    throw new RuntimeException("Invalid activation rank");
            }

            return new Pair((Object)null, new INDArray[]{out});
        }
    }

    public String toString() {
        return "AverageVertex(id=" + this.getVertexIndex() + ",name=\"" + this.getVertexName() + "\")";
    }

    public void setBackpropGradientsViewArray(INDArray backpropGradientsViewArray) {
        if (backpropGradientsViewArray != null) {
            throw new RuntimeException("Vertex does not have gradients; gradients view array cannot be set here");
        }
    }

    public Pair<INDArray, MaskState> feedForwardMaskArrays(INDArray[] maskArrays, MaskState currentMaskState, int minibatchSize) {
        return maskArrays != null && maskArrays.length != 0 ? new Pair(maskArrays[0], currentMaskState) : null;
    }
}

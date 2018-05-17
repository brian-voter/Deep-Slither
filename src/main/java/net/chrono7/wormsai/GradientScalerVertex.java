//
// Source code recreated from a .class file by IntelliJ IDEA
// (powered by Fernflower decompiler)
//

package net.chrono7.wormsai;

import org.deeplearning4j.nn.conf.graph.GraphVertex;
import org.deeplearning4j.nn.conf.inputs.InputType;
import org.deeplearning4j.nn.conf.inputs.InvalidInputTypeException;
import org.deeplearning4j.nn.conf.memory.LayerMemoryReport.Builder;
import org.deeplearning4j.nn.conf.memory.MemoryReport;
import org.deeplearning4j.nn.graph.ComputationGraph;
import org.nd4j.linalg.api.ndarray.INDArray;

import java.util.Arrays;

public class GradientScalerVertex extends GraphVertex {

    public GradientScalerVertex clone() {
        return new GradientScalerVertex();
    }

    public boolean equals(Object o) {
        if (!(o instanceof GradientScalerVertex)) {
            return false;
        } else {
            GradientScalerVertex s = (GradientScalerVertex) o;
            return true;
        }
    }

    public int hashCode() {
        return 12;
    }

    public int numParams(boolean backprop) {
        return 0;
    }

    public int minVertexInputs() {
        return 1;
    }

    public int maxVertexInputs() {
        return 1;
    }

    public org.deeplearning4j.nn.graph.vertex.GraphVertex instantiate(ComputationGraph graph, String name, int idx, INDArray paramsView, boolean initializeParams) {
        return new GradientScalerVertexImpl(graph, name, idx);
    }

    public InputType getOutputType(int layerIndex, InputType... vertexInputs) throws InvalidInputTypeException {
        if (vertexInputs.length != 1) {
            throw new InvalidInputTypeException("GradientScalerVertex expects single input type. Received: " + Arrays.toString(vertexInputs));
        } else {
            switch (vertexInputs[0].getType()) {
                case FF:
                    return InputType.feedForward(vertexInputs[0].arrayElementsPerExample());
                case CNN:
                    return vertexInputs[0];
                default:
                    throw new RuntimeException("Only supports FF");
            }
        }
    }

    public MemoryReport getMemoryReport(InputType... inputTypes) {
        InputType outputType = this.getOutputType(-1, inputTypes);
        return (new Builder((String) null, GradientScalerVertex.class, inputTypes[0], outputType)).standardMemory(0L, 0L).workingMemory(0L, 0L, 0L, 0L).cacheMemory(0L, 0L).build();
    }

    public String toString() {
        return "GradientScalerVertex()";
    }
}

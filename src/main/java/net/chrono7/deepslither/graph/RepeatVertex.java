//
// Source code recreated from a .class file by IntelliJ IDEA
// (powered by Fernflower decompiler)
//

package net.chrono7.deepslither.graph;

import org.deeplearning4j.nn.conf.graph.GraphVertex;
import org.deeplearning4j.nn.conf.inputs.InputType;
import org.deeplearning4j.nn.conf.inputs.InvalidInputTypeException;
import org.deeplearning4j.nn.conf.memory.LayerMemoryReport.Builder;
import org.deeplearning4j.nn.conf.memory.MemoryReport;
import org.deeplearning4j.nn.graph.ComputationGraph;
import org.nd4j.linalg.api.ndarray.INDArray;

import java.util.Arrays;

public class RepeatVertex extends GraphVertex {
    private int nOut;

    public RepeatVertex(int nOut) {
        this.nOut = nOut;
    }

    public RepeatVertex clone() {
        return new RepeatVertex(nOut);
    }

    public boolean equals(Object o) {
        if (!(o instanceof RepeatVertex)) {
            return false;
        } else {
            RepeatVertex s = (RepeatVertex)o;
            return s.nOut == this.nOut;
        }
    }

    public int hashCode() {
        return (new Integer(this.nOut)).hashCode();
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
        return new RepeatVertexImpl(graph, name, idx, this.nOut);
    }

    public InputType getOutputType(int layerIndex, InputType... vertexInputs) throws InvalidInputTypeException {
        if (vertexInputs.length != 1) {
            throw new InvalidInputTypeException("SubsetVertex expects single input type. Received: " + Arrays.toString(vertexInputs));
        } else {
            switch(vertexInputs[0].getType()) {
                case FF:
                    return InputType.feedForward(nOut);
                default:
                    throw new RuntimeException("Only supports FF");
            }
        }
    }

    public MemoryReport getMemoryReport(InputType... inputTypes) {
        InputType outputType = this.getOutputType(-1, inputTypes);
        return (new Builder((String)null, RepeatVertex.class, inputTypes[0], outputType)).standardMemory(0L, 0L).workingMemory(0L, 0L, 0L, 0L).cacheMemory(0L, 0L).build();
    }

    public int getNOut() { return this.nOut; }

    public void setNOut(int nOut) {
        this.nOut = nOut;
    }

    public String toString() {
        return "ReplicateVertex(nOut=" + nOut + ")";
    }
}

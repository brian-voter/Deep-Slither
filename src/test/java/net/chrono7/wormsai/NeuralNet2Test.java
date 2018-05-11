//package net.chrono7.wormsai;
//
//import org.junit.jupiter.api.Test;
//import org.nd4j.linalg.api.ndarray.INDArray;
//import org.nd4j.linalg.factory.Nd4j;
//
//import static org.junit.jupiter.api.Assertions.*;
//
//class NeuralNet2Test {
//
//    @Test
//    void rowToMatrix() {
//        INDArray test1 = Nd4j.create(new float[]{1,2});
//        INDArray test1Expected = Nd4j.create(new float[]{1,2,1,2,1,2}, new int[]{3, 2});
//
//        System.out.println(test1);
//        System.out.println(test1Expected);
//
//        assert NeuralNet2.rowToMatrix(test1, 3).equals(test1Expected);
//        assert NeuralNet2.rowToMatrix(test1, 1).equals(test1);
//    }
//}
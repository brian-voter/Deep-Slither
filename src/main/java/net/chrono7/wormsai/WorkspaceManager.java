package net.chrono7.wormsai;

import org.nd4j.linalg.api.memory.conf.WorkspaceConfiguration;
import org.nd4j.linalg.api.memory.enums.AllocationPolicy;
import org.nd4j.linalg.api.memory.enums.LearningPolicy;
import org.nd4j.linalg.api.memory.enums.MirroringPolicy;
import org.nd4j.linalg.api.memory.enums.ResetPolicy;

public class WorkspaceManager {

    public static final WorkspaceConfiguration cpuConfig;
    public static final WorkspaceConfiguration gpuConfig;
    public static final String CPU_ID = "CPU_WORKSPACE";
    public static final String GPU_ID = "GPU_WORKSPACE";
    private static final long CPU_BUFFER_SIZE = 10 * (10^9);
    private static final long GPU_BUFFER_SIZE = 4 * (10^9);

    static {
        cpuConfig = WorkspaceConfiguration.builder()
                .initialSize(CPU_BUFFER_SIZE)
                .policyAllocation(AllocationPolicy.STRICT)
                .policyLearning(LearningPolicy.NONE)
                .policyMirroring(MirroringPolicy.HOST_ONLY)
                .policyReset(ResetPolicy.ENDOFBUFFER_REACHED) // <--- this option makes workspace act as circular buffer, beware.
                .build();

        gpuConfig = WorkspaceConfiguration.builder()
                .initialSize(GPU_BUFFER_SIZE)
                .policyAllocation(AllocationPolicy.STRICT)
                .policyLearning(LearningPolicy.NONE)
                .policyReset(ResetPolicy.ENDOFBUFFER_REACHED) // <--- this option makes workspace act as circular buffer, beware.
                .build();
    }

}

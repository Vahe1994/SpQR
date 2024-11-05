#!/bin/bash

export NCU_EXPORT_PATH=
export NCU_PATH=
export COMMAND=

CUDA_VISIBLE_DEVICES=0 $NCU_PATH --import-source yes --export $NCU_EXPORT_PATH --force-overwrite --section ComputeWorkloadAnalysis --section InstructionStats --section LaunchStats --section MemoryWorkloadAnalysis --section MemoryWorkloadAnalysis_Chart --section MemoryWorkloadAnalysis_Tables --section Nvlink --section Nvlink_Tables --section Nvlink_Topology --section Occupancy --section SchedulerStats --section SourceCounters --section SpeedOfLight --section SpeedOfLight_HierarchicalDoubleRooflineChart --section SpeedOfLight_HierarchicalHalfRooflineChart --section SpeedOfLight_HierarchicalSingleRooflineChart --section SpeedOfLight_HierarchicalTensorRooflineChart --section SpeedOfLight_RooflineChart --section WarpStateStats $COMMAND


# Z_control_multi_agents
 Z-Control for consensus in High-Order Multi-Agent Systems

Angela Monti, Institute for Applied Mathematics (IAC), CNR, Bari, Italy. Mail: angela.monti@cnr.it

Z-control is a MATLAB routine (tested with version R2024b) designed for consensus in high-order multi-agent systems via Z-control. The codes implement different Z-control strategies for consensus.

The repository contains:
Opinion\_1st\_no\_control.m: opinion dynamics without control.
Opinion\_1st\_zcontrol.m: opinion dynamics under Z-control.

CS\_2nd\_no\_control.m: simulation of second-order agents without control.
CS\_2nd\_z\_control\_direct.m: implementation of direct Z-control for second-order agents.
CS\_2nd\_z\_control\_ind.m: implementation of indirect Z-control for second-order agents.

CS\_3rd\_nocontrol.m: simulation of third-order agents without control.
CS\_3rd\_z\_control\_direct.m: direct Z-control for third-order agents.
CS\_3rd\_z\_control\_ind\_pos.m: indirect Z-control through on positions.
CS\_3rd\_z\_control\_ind\_vel.m: indirect Z-control through on velocities.

Data files (.mat): datasets for reproducibility of experiments.

The routines have been implemented and developed by Angela Monti and Fasma Diele. They can be used under the conditions of CC-BY-NC 2.0. When utilizing this codebase, please cite the following publication:

A. Monti, F. Diele, Exponential Consensus through Z-Control in High-Order Multi-Agent Systems, arXiv preprint.

The complete description of the model, control architecture, and numerical methods is available in the cited manuscript.

The development and implementation of the model and routines have been made possible thanks to the National Recovery and Resilience Plan (NRRP), Mission 4 Component 2 Investment 1.4—Call for tender No. 3138 of 16 December 2021, rectified by Decree No. 3175 of 18 December 2021 of the Italian Ministry of University and Research, funded by the European Union—NextGenerationEU; Award Number: Project code CN 00000033, Concession Decree No. 1034 of 17 June 2022 adopted by the Italian Ministry of University and Research, CUP B83C22002930006, Project title “National Biodiversity Future Center” (NBFC).

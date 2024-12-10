1. Introduction to the Enhanced Evaluation Section

Introduction

- State the motivation based on the research gap identified in the taxonomy of online process mining methods.
- Highlight the limitations of existing window-based methods, such as susceptibility to concept drift, finding sub-optimal windows, and the need for parameter process knowledge.
- Introduce your approach as a solution to these limitations, focusing on its strengths in handling concept drift and finding optimal windows without extensive process knowledge.

2. Evaluation Goals Integration

Evaluation Goals from Ceravolo et al. (2022)

- Minimize Memory Consumption (G1)
- Minimize Response Latency (G2)
- Minimize the Number of Runs (G3)
- Optimize Accuracy (G4)

3. Detailed Evaluation Methods and Results

A. Minimize Memory Consumption (G1)

- Integration: Compare the memory usage of our method with CountBasedWindow and LandmarkWindow.
- Method: Measure the average and peak memory consumption during the evaluation period.
- Results and Discussion: Acknowledge that CountBased and LandmarkWindows might have better results in memory consumption due to their simplicity. Present our method's memory usage data and discuss its efficiency in handling complex scenarios despite potentially higher memory usage.

B. Minimize Response Latency (G2)

- Integration: Evaluate the processing latency of our method versus CountBasedWindow and LandmarkWindow.
- Method: Record the time taken to process events and update the process model for each method.
- Results and Discussion: Acknowledge that CountBasedWindow and LandmarkWindow might have lower latency due to their straightforward operations. Highlight that our method's slightly higher latency is justified by its ability to handle concept drift and optimize window sizes dynamically.

C. Minimize the Number of Runs (G3)

- Integration: Determine the frequency of window adjustments required by each method.
- Method: Track and count the number of window adjustments over the evaluation period.
- Results and Discussion: Show that our method requires fewer adjustments due to its dynamic nature, which reduces the computational overhead in the long run. Use tables or graphs to visualize the frequency of window changes.

D. Optimize Accuracy (G4)

- Integration: Measure the accuracy of process models derived from different methods using fitness and precision metrics.
- Method: Evaluate the fitness and precision of process models.
- Results and Discussion: Present accuracy results for each method, showing how our approach maintains high accuracy. Use bar charts or precision-recall curves to compare the methods and highlight the superior accuracy of our approach in capturing the true process model.

E. Handling Concept Drift

- Integration: Assess the robustness of each method to concept drift.
- Method: Introduce synthetic concept drifts in the event streams and evaluate how each method adapts.
- Results and Discussion: Highlight the resilience of our method to concept drift compared to others. Use case studies or scenario analysis to illustrate the handling of concept drift, showing clear advantages of our approach.

4. Conclusion

Conclusion

- Summarize the key findings from the evaluation, emphasizing how our method addresses the limitations of existing approaches while recognizing areas where they might excel.
- Reiterate the benefits of our method in terms of handling concept drift and optimizing window sizes without extensive process knowledge.
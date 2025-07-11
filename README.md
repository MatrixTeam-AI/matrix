# The Matrix 
<div align="center">
<img src=readme_src/white-logo.svg width="50%"/>
</div>
<p align="center">
Download The Matrix model weights at <a href="https://huggingface.co/MatrixTeam/TheMatrix" target="_blank"> 🤗 Huggingface</a> or <a href="https://www.modelscope.cn/models/AiurRuili/TheMatrix" target="_blank"> 🤖 ModelScope</a>
</p>
<p align="center">
Download The Matrix Dataset at <a href="https://huggingface.co" target="_blank"> 🤗 Huggingface</a> or <a href="https://www.modelscope.cn/datasets/TheMatrixDataset/the_matrix_dataset_8M_1920_1080" target="_blank"> 🤖 ModelScope</a>
</p>
<p align="center">
📚 View the <a href="https://arxiv.org/abs/2412.03568" target="_blank">Paper</a>, <a href="https://matrixteam-ai.github.io/pages/TheMatrix/" target="_blank"> Website, and <a href="http://matrixteam-ai.github.io/docs/TheMatrixDocs" target="_blank">Documentation</a>
</p>
<p align="center">
    👋 Say Hi to our team and members at <a href="https://matrixteam-ai.github.io/" target="_blank">Matrix-Team</a>
</p>
<p align="center">
📍 Explore The Matrix playground online at <a href="">Journee</a> to experience real-time AI generated world.
</p>
<p align="center">
  Real-Time Inference Tech Report: <a href="coming-soon.html" target="_blank">Coming Soon</a>
</p>

<h2 align="center">🌍 Global Adoption &amp; Acknowledgements</h2>

<p align="center">
  The <strong>Matrix&nbsp;Dataset</strong> has already been cited, tested, or deployed by the following research labs and industry teams.<br/>
  Thank you for advancing open, multimodal research together!
</p>

<div align="center">
  <table style="border-collapse:collapse;font-size:90%;line-height:1.4em;">
    <tr>
      <td align="center">🧪<br/>Dynamics&nbsp;Lab</td>
      <td align="center">🏫<br/>UC&nbsp;San&nbsp;Diego</td>
      <td align="center">🏫<br/>University&nbsp;of&nbsp;Waterloo</td>
      <td align="center">🏢<br/>Microsoft&nbsp;Research</td>
    </tr>
    <tr>
      <td align="center">🏢<br/>ByteDance</td>
      <td align="center">🏢<br/>Alibaba</td>
      <td align="center">🏢<br/>Tencent</td>
      <td align="center">🤖<br/>Duality&nbsp;AI</td>
    </tr>
    <tr>
      <td align="center">💻<br/>AMD&nbsp;AI&nbsp;Group</td>
      <td align="center">🕹️<br/>Journee</td>
      <td align="center">🏫<br/>University&nbsp;of&nbsp;Hong&nbsp;Kong</td>
      <td align="center">🏫<br/>Nanyang&nbsp;Technological&nbsp;University</td>
    </tr>
    <tr>
      <td align="center">🏫<br/>Vector&nbsp;Institute</td>
      <td align="center">🏫<br/>Hong&nbsp;Kong&nbsp;UST</td>
      <td align="center">🏫<br/>Tsinghua&nbsp;University</td>
      <td align="center">✨<br/>…and&nbsp;growing!</td>
    </tr>
  </table>
</div>


---
## What is The Matrix?
The Matrix is an advanced world model designed to generate **high-quality, infinite-time interactive videos in real-time**, setting a new benchmark in the field of **neural interactive simulations**. It is simultaneously several innovations:

- **A cutting-edge world model** that generates continuous, interactive video content with **unparalleled realism and length**.
- A **real-time** system that supports infinite content generation, overcoming previous limitations seen in simpler 2D game models like DOOM or Minecraft.
- A powerful model architecture powered by the **Swin-DPM** model, designed to produce dynamic, **ever-expanding** content.
- A novel training strategy that integrates both real and simulated data, enhancing the system's ability to exceptional **generalization** capabilities.

At its core, The Matrix combines these elements to push the boundaries of interactive video generation, making real-time, high-quality, infinite-length content a reality.

## Documentation
Comprehensive documentation is available in [English](). This includes detailed installation steps, tutorials, and training instructions. The [paper](https://arxiv.org/abs/2412.03568) and [Project Page](https://matrixteam-ai.github.io/pages/TheMatrix/) offer more details about the method.


## Model Weights
Model checkpoints can be found in [Huggingface](https://huggingface.co/MatrixTeam/TheMatrix) and [ModelScope](https://www.modelscope.cn/models/AiurRuili/TheMatrix). Please refer to the [Documentation](http://matrixteam-ai.github.io/docs/TheMatrixDocs) for how to load them for inferences.


## Important Updates

According to a request from Alibaba Tongyi Lab, the previous version of The Matrix was inherited from an internal version of Video DiT and could not be openly released. Therefore, we have re-implemented The Matrix code based on the previously open-released video generation model, [CogVideoX](https://github.com/THUDM/CogVideo/tree/main). We sincerely appreciate the efforts of the **CogVideo** team for their contributions.
### Implemented Features

Most planned components are now live, delivering real-time, infinite-horizon generation at **16 FPS** with near-zero latency:

- ✅ **8-GPU Parallel Inference for DiT & VAE**  
  Both the Diffusion Transformer (DiT) backbone and our VAE decoder run across 8 GPUs in parallel, yielding a **6–8× speedup** over single-GPU baselines.

- ✅ **Stream Consistency Models**  
  Advanced consistency losses enable uninterrupted generation over arbitrary lengths, boosting end-to-end throughput by **7–10×**.

### Key Capabilities

- **Real-Time Control**  
  Instantly respond to live inputs (e.g., steering, throttle), updating the generated scene in ** real time**.

- **Infinite-Horizon Generation**  
  Seamlessly extend scenes without drift or degradation—generate as long as you like.

- **Low-Latency Feedback Loop**  
  End-to-end system sustains a continuous **16 FPS** render/playback cycle for smooth interactive experiences.
  
### Latency

| GPU Type | Latency | VAEnum | DiTnum |
|----------|---------|--------|--------|
| A100     | 0.6 s   |    3   |    5   |
| A900     | 0.6 s   |    3   |    5   |
| L40      | 1.2 s   |    1   |    3   |
| H100     |   -     |    -   |    -   |


### Known Issues

- **Latency bottlenecks**  
   End-to-end inference sometimes falls below real-time requirements under heavy load.

- **Color degradation on long straight segments**  
   Sustained straight driving causes gradual visual drift; sharp turns temporarily correct colors.

- **Global consistency drift**  
   Over extended horizons, scene coherence can degrade (e.g., object placement, lighting).

### Planned

- [ ] **Training on Fused Realistic + Simulated Data**  
  Joint training on real-world captures and high-fidelity simulations to acquire stronger generalization ability.

- [ ] **Latency optimization**  
  Profiling and kernel fusion to further reduce end-to-end inference time.

- [ ] **Color stability enhancement**  
  Incorporate temporal color correction modules to prevent degradation on straight paths.

- [ ] **Global consistency models**  
  Develop long-range consistency losses and memory mechanisms to maintain scene coherence indefinitely.

## Reimplementation contributions
The successful release of The Matrix Project is built upon the collective efforts of our incredibly talented team members. We extend our heartfelt gratitude for their dedication, hard work, and invaluable contributions. Those members are:

**Longxiang Tang, Zhicai Wang, Ruili Feng, Ruihang Chu, Han Zhang, and Zhantao Yang**

Special Thanks to **Longxiang** and **Zhicai** for their excellent contributions.

## Additional Notes

There have been certain changes to the hyperparameter settings and training strategy compared to what is reported in the paper due to the re-implementation. Please be aware of these when reviewing the code.

Despite these changes, we are pleased to announce that the overall generation quality is much more advanced compared to the previous version after more careful design of methods and parameters.

## Citation
If you find our work useful please consider citing:

```bibtex
@article{feng2024matrix,
  title={The matrix: Infinite-horizon world generation with real-time moving control},
  author={Feng, Ruili and Zhang, Han and Yang, Zhantao and Xiao, Jie and Shu, Zhilei and Liu, Zhiheng and Zheng, Andy and Huang, Yukun and Liu, Yu and Zhang, Hongyang},
  journal={arXiv preprint arXiv:2412.03568},
  year={2024}
}
```

## License
The code in this repository is released under the Apache 2.0 License.

The Matrix model (including its corresponding Transformers module and VAE module) is released under the Apache 2.0 License.

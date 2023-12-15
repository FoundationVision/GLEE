# GLEE: General Object Foundation Model for Images and Videos at Scale
> #### Junfeng Wu\*, Yi Jiang\*,  Qihao Liu, Zehuan Yuan, Xiang Bai<sup>&dagger;</sup>,and Song Bai<sup>&dagger;</sup>
>
> \* Equal Contribution, <sup>&dagger;</sup>Correspondence

\[[Project Page](https://glee-vision.github.io/)\]   \[[Paper](https://arxiv.org/pdf/.pdf)\]    \[[HuggingFace Demo](https://huggingface.co/spaces/Junfeng5/GLEE_demo)\] 

![data_demo](assets/images/data_demo.png)

**GLEE is a general object foundation model jointly trained on over ten million images from various benchmarks with diverse levels of supervision, which excels in a wide range of object-centric tasks while maintaining SOTA performance. It also showcases remarkable versatility and robust zero-shot transferability.**



We will release the following contents for **GLEE**:exclamation:

- [x] Demo Code
- [x] Model Checkpoint
- [ ] Comprehensive User Guide
- [ ] Training Code
- [ ] Evaluation Code



## Run the demo APP

Try our online demo app on \[[HuggingFace Demo](https://huggingface.co/spaces/Junfeng5/GLEE_demo)\] or use it locally:

```bash
git clone https://github.com/FoundationVision/GLEE
cd GLEE/app/
pip install -r requirements.txt
```

Download the pretrain weight for [GLEE-Lite](https://huggingface.co/spaces/Junfeng5/GLEE_demo/resolve/main/GLEE_R50_Scaleup10m.pth?download=true) and [GLEE-Plus](https://huggingface.co/spaces/Junfeng5/GLEE_demo/resolve/main/GLEE_SwinL_Scaleup10m.pth?download=true) 

```
# support CPU and GPU running
python app.py
```









# Citation

```
@misc{wu2023GLEE,
  author= {Junfeng Wu, Yi Jiang, Qihao Liu, Zehuan Yuan, Xiang Bai, Song Bai},
  title = {General Object Foundation Model for Images and Videos at Scale},
  year={2023},
  eprint={2312.09158},
  archivePrefix={arXiv}
}
```

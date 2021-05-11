# [Robustness to Missing Features using Hierarchical Clustering with Split Neural Networks](https://arxiv.org/abs/2011.09596)

Code for the paper - [https://arxiv.org/abs/2011.09596](https://arxiv.org/abs/2011.09596)

The following readme has simple steps to reproduce the training and evaluation for any of the datasets mentioned.

## Setup
1. Setup Virtual Environment
```
pip install virtualenv
virtualenv venv
source venv/bin/activate
```
2. Install dependencies
`pip install -r requirements.txt`

3. Run the code

## Run

### Train
```
python main.py train --dataset_dir datasets --dataset life --model_dir life_models --verbose 1
```

### Evaluate
```
python main.py evaluate --dataset_dir datasets --dataset life --model_dir life_models --verbose 1
```

## Further Notes

### Mapping for datasets to `--dataset` flag

1. Life Expectancy (WHO) : life
2. Bands : bands
3. Kidney Disease : kidney_disease
4. Mammographics : mammographics
5. Horse Colic : horse
6. Pima Indians : pima
7. Hepatitis : hepatitis
8. Breast Cancer Winconsin : winconsin

## Citation

If you find this project useful for your research, please use the following BibTeX entry to cite our paper [https://arxiv.org/abs/2011.09596](https://arxiv.org/abs/2011.09596).

    @misc{khincha2020robustness,
          title={Robustness to Missing Features using Hierarchical Clustering with Split Neural Networks}, 
          author={Rishab Khincha and Utkarsh Sarawgi and Wazeer Zulfikar and Pattie Maes},
          year={2020},
          eprint={2011.09596},
          archivePrefix={arXiv},
          primaryClass={cs.LG}
    }

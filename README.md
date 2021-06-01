# Robustness to Missing Features using Hierarchical Clustering with Split Neural Networks

Code for the paper - [https://ojs.aaai.org/index.php/AAAI/article/view/17905](https://ojs.aaai.org/index.php/AAAI/article/view/17905)

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

If you find this project useful for your research, please use the following BibTeX entry to cite our paper [https://ojs.aaai.org/index.php/AAAI/article/view/17905](https://ojs.aaai.org/index.php/AAAI/article/view/17905).

    @article{khincha2021missing,
            author={Khincha, Rishab and Sarawgi, Utkarsh and Zulfikar, Wazeer and Maes, Pattie}, 
            title={Robustness to Missing Features using Hierarchical Clustering with Split Neural Networks (Student Abstract)}, 
            volume={35}, 
            url={https://ojs.aaai.org/index.php/AAAI/article/view/17905}, 
            number={18}, 
            journal={Proceedings of the AAAI Conference on Artificial Intelligence}, 
            year={2021}, 
            month={May}, 
            pages={15817-15818}
    }

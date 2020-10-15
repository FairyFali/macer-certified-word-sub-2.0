### 最大化可信边界2.0

#### Bert IMDB Normal Train
epoch 0: eval result acc=0.8976 loss=0.28
epoch 1: eval result acc=0.8868 loss=0.44
epoch 2: eval result acc=0.8788 loss=0.51

python train.py --data_dir ../data/aclImdb --model_type bert --task_name imdb --batch_size 16 --train --from_scratch
python train.py --data_dir ../data/aclImdb --model_type bert --task_name imdb --batch_size 16 --eval --ckpt 0
python train.py --data_dir ../data/aclImdb --model_type bert --task_name imdb --batch_size 16 --eval --train_type rs --ckpt 7

BERT训练超过4个epoch，准确率极速下降为0.5

#### CNN IMDB Normal Train
eval result acc=0.8532 loss=0.43 epoch=19

!python train.py --data_dir ../data/aclImdb --model_type cnn --task_name imdb --batch_size 32 --train
!python train.py --data_dir ../data/aclImdb --model_type cnn --task_name imdb --batch_size 32 --eval --ckpt 19

#### LSTM IMDB Normal Train
eval result acc=0.8439 loss=0.55 epoch 12

!python train.py --data_dir ../data/aclImdb --model_type lstm --task_name imdb --batch_size 32 --train
!python train.py --data_dir ../data/aclImdb --model_type lstm --task_name imdb --batch_size 32 --eval --ckpt 12

#### CharCNN IMDB Normal Train
0.8460 epoch=21

python train.py --data_dir ../data/aclImdb --model_type char-cnn --task_name imdb --batch_size 32 --max_grad_norm 400 --num_train_epochs 60 --train --from_scratch
python train.py --data_dir ../data/aclImdb --model_type char-cnn --task_name imdb --batch_size 32 --eval --ckpt 21

+ IMDB数据集包含了50000条偏向明显的评论，其中25000条作为训练集，25000作为测试集。label为pos(positive)和neg(negative)。

#### Amazon Normal Train
+ 自己的PC跑不起来，内存不够
CUDA_VISIBLE_DEVICES=2 python train.py --data_dir ../data/amazon_review_full_csv --model_type bert --task_name amazon --max_seq_length 256 --batch_size 16 --train --from_scratch
python train.py --data_dir ../data/amazon_review_full_csv --model_type char-cnn --task_name amazon --batch_size 16 --eval --ckpt 3
17h/epoch
epoch 0 0.6036
epoch 3 0.6095

python train.py --data_dir ../data/amazon_review_full_csv --model_type cnn --task_name amazon --max_seq_length 256 --batch_size 32 --train
python train.py --data_dir ../data/amazon_review_full_csv --model_type cnn --task_name amazon --max_seq_length 256 --batch_size 32 --eval --ckpt 19
epoch 19 0.5779

python train.py --data_dir ../data/amazon_review_full_csv --model_type lstm --task_name amazon --max_seq_length 256 --batch_size 32 --train
3h/epoch
python train.py --data_dir ../data/amazon_review_full_csv --model_type lstm --task_name amazon --max_seq_length 256 --batch_size 32 --eval --ckpt 12
10min/次
ckpt=3 0.5941
epoch 12 

python train.py --data_dir ../data/amazon_review_full_csv --model_type char-cnn --task_name amazon --batch_size 32 --train --from_scratch

+ 亚马逊评论全得分数据集是通过从1到5的每个评论得分中随机抽取600,000（60w）个训练样本和130,000（13w）个测试样本来构建的。总共有3,000,000（300w）个Trainig样本和650,000（65w）个测试样本。

#### Data Preprocess
python data_preprocess.py --task_name imdb --data_dir ../data/aclImdb --embed_dir ../data
python data_preprocess.py --task_name amazon --data_dir ../data/amazon_review_full_csv --embed_dir ../data

#### Random smooth train
python random_smooth_train.py --data_dir ../data/aclImdb --model_type bert --task_name imdb --batch_size 16 --pop_size 32 --train --from_scratch --learning_rate 1e-6

python random_smooth_train.py --data_dir ../data/aclImdb --model_type cnn --task_name imdb --batch_size 32 --train --from_scratch 
9 0.8534
10 0.8542
11 0.8531
12 0.8531
13 0.8530
python random_smooth_train.py --data_dir ../data/aclImdb --model_type char-cnn --task_name imdb --batch_size 32 --max_grad_norm 400 --train --from_scratch --learning_rate 1e-6
python random_smooth_train.py --data_dir ../data/aclImdb --model_type lstm --task_name imdb --batch_size 32 --train --from_scratch --learning_rate 1e-6

#### Evaluate
##### Normal
python evaluation.py --skip 100 --num_random_sample 5000 --mc_error 0.01 --task_name imdb --model_type bert --data_dir ../data/aclImdb --batch_size 16 --max_seq_length 256 --ckpt 0

python evaluation.py --skip 100 --num_random_sample 5000 --mc_error 0.01 --task_name imdb --model_type cnn --data_dir ../data/aclImdb --batch_size 32 --max_seq_length 256 --ckpt 19

python evaluation.py --skip 100 --num_random_sample 5000 --mc_error 0.01 --task_name imdb --model_type char-cnn --data_dir ../data/aclImdb --batch_size 32 --ckpt 21
0.828 4h
python evaluation.py --skip 100 --num_random_sample 5000 --mc_error 0.01 --task_name imdb --model_type lstm --data_dir ../data/aclImdb --batch_size 32 --max_seq_length 256 --ckpt 12

##### RS
+ CNN
python evaluation.py --skip 100 --num_random_sample 5000 --mc_error 0.01 --task_name imdb --model_type cnn --data_dir ../data/aclImdb --batch_size 32 --max_seq_length 256 --train_type rs --ckpt 10
0.86

+ BERT
python evaluation.py --skip 100 --num_random_sample 5000 --mc_error 0.01 --task_name imdb --model_type bert --data_dir ../data/aclImdb --batch_size 16 --max_seq_length 256 --train_type rs --ckpt 7

+ LSTM
python evaluation.py --skip 100 --num_random_sample 5000 --mc_error 0.01 --task_name imdb --model_type lstm --data_dir ../data/aclImdb --batch_size 32 --max_seq_length 256 --train_type rs --ckpt 31
0.836

+ others
python evaluation.py --skip 1000 --num_random_sample 5000 --mc_error 0.01 --task_name amazon --model_type cnn --data_dir ../data/amazon_review_full_csv --batch_size 32 --max_seq_length 256 --ckpt 19
0.5261538461538462

IMDB:

|  | Accuracy_clean | Accuracy_certification | |
| :-----| ----: | :----: | :----: |
| LSTM-Normal | 84.39 | 81.60 | |
| CNN-Normal | 85.32 | 83.60 | |
| CharCNN-Normal | 84.60 | 82.80 | |
| BERT-Normal | 90.38 | 87.20/86.00 | |
| LSTM-RS(31) | 85.85 | 84.40 | |
| CNN-RS | 85.42 | 86.00 | |
| BERT-RS | 90.80 | 87.20 | |

Amazon

### SNLI
#### Normal Train
+ BOW 
python entailment_train_evaluate.py --data_dir ../data/snli_1.0 --model_type bow --task_name snli --batch_size 32 --max_seq_length 32 --train --from_scratch
25 0.7942

+ DecompAtten 不用了,复现不出来
python entailment_train_evaluate.py --data_dir ../data/snli_1.0 --model_type decom_att --task_name snli --batch_size 32 --max_seq_length 32 --learning_rate 5e-4 --weight_decay 1e-4 --train --from_scratch
63.85

+ esim
CUDA_VISIBLE_DEVICES=2 python entailment_train_evaluate.py --data_dir ../data/snli_1.0 --model_type esim --task_name snli --batch_size 32 --max_seq_length 32 --train --from_scratch
epoch=15, acc=0.8562

+ BERT
python entailment_train_evaluate.py --data_dir ../data/snli_1.0 --model_type bert --task_name snli --batch_size 32 --train --from_scratch
2 0.8932

#### RS Evaluate
+ Preprocess

python data_preprocess.py --task_name snli --data_dir ../data/snli_1.0 --embed_dir ../data

+ eval

python entailment_rs_evaluation.py --skip 100 --num_random_sample 500 --mc_error 0.01 --task_name snli --model_type bow --data_dir ../data/snli_1.0 --batch_size 32 --max_seq_length 32 --ckpt 25
0.75

python entailment_rs_evaluation.py --skip 100 --num_random_sample 500 --mc_error 0.01 --task_name snli --model_type esim --data_dir ../data/snli_1.0 --batch_size 32 --max_seq_length 32 --ckpt 15
0.8

python entailment_rs_evaluation.py --skip 100 --num_random_sample 500 --mc_error 0.01 --task_name snli --model_type bert --data_dir ../data/snli_1.0 --batch_size 32 --ckpt 2
0.86

#### RS Train
+ BOW
python entailment_rs_train.py --data_dir ../data/snli_1.0 --model_type bow --task_name snli --batch_size 32 --max_seq_length 32 --train --from_scratch --learning_rate 1e-4
eval 9 0.7881 8 0.7886 7 0.7873 7_ 0.7967 6 0.7892 5 0.7914 4 0.7920 3 0.7904
python entailment_train_evaluate.py --data_dir ../data/snli_1.0 --model_type bow --task_name snli --batch_size 32 --max_seq_length 32 --train_type rs --eval --ckpt 7
test 0.7967
python entailment_rs_evaluation.py --skip 100 --num_random_sample 500 --mc_error 0.01 --task_name snli --model_type bow --data_dir ../data/snli_1.0 --batch_size 32 --max_seq_length 32 --train_type rs --ckpt 7
rs test
0.77
+ ESIM
python entailment_rs_train.py --data_dir ../data/snli_1.0 --model_type esim --task_name snli --batch_size 32 --max_seq_length 32 --train --from_scratch --learning_rate 1e-4

python entailment_train_evaluate.py --data_dir ../data/snli_1.0 --model_type esim --task_name snli --batch_size 32 --max_seq_length 32 --train_type rs --eval --ckpt 4
test 0.8530
python entailment_rs_evaluation.py --skip 100 --num_random_sample 500 --mc_error 0.01 --task_name snli --model_type esim --data_dir ../data/snli_1.0 --batch_size 32 --max_seq_length 32 --train_type rs --ckpt 4
rs test 0.87

+ BERT
python entailment_rs_train.py --data_dir ../data/snli_1.0 --model_type bert --task_name snli --batch_size 32 --pop_size 8 --max_seq_length 128 --train --from_scratch --learning_rate 1e-7

| SNLI | Accuracy_clean | Accuracy_certification(100) | |
| :-----| ----: | :----: | :----: |
| BOW-Normal | 79.42 | 75.0 | |
| ESIM-Normal | 85.62 | 80.0 | |
| BERT-Normal | 89.32 | 86.0 | |
| BOW-RS | 79.67 | 77.00 | |
| ESIM-RS | 85.30 | 87.0 | |
| BERT-RS |  |  | |

#### 阅读论文
+ Generating Natural Language Adversarial Examples, Alzantot
+ Certified Robustness to Adversarial Word Substitutions, Jia
+ Character-level convolutional networks for text classification. Zhang 2015, NIPS
+ Is BERT Really Robust? A Strong Baseline for Natural Language Attack on Text Classification and Entailment
+ Defense against Adversarial Attacks in NLP via Dirichlet Neighborhood Ensemble, 2020 预印版
+ 写作借鉴IBP和DNE

#### idea
+ 借鉴于黄萱菁的论文的思路，测试攻击时的扰动集合应该小于训练攻击时的扰动集合，所以这篇文章的思路可以称之为Muti-Hop邻居



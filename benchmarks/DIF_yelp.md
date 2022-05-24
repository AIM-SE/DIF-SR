## DIF_Amazon_Beauty

A notebook to benchmark DIF-SR on yelp dataset.

Author: Peilin Zhou, Upstage


### Index
[Environments](#Environments) | [Dataset](#Dataset) | [Code](#Code) | [Results](#Results) | [Logs](#Logs)

### Environments
+ Hardware

    ```bash
    CPU: AMD Ryzen Threadripper 3970X 32-Core Processor
    RAM: 256
    GPU: NVIDIA GeForce RTX 3090
    ```
+ Software

    ```python
    python: 3.8.0
    pytorch 1.8.1
    numpy: 1.20.3
    scipy: 1.6.3
    tensorboard: 2.7.0
    ```

### Dataset
Following commonly used pre-processing steps in SR,
we remove all items and users that occur less than five times in
these datasets. All the interactions are regarded as implicit feedback.

You could download datasets from [RecSysDatasets](https://github.com/RUCAIBox/RecSysDatasets) or their [Google Drive](https://drive.google.com/drive/folders/1ahiLmzU7cGRPXf5qGMqtAChte2eYp9gI). 
### Code

Our implementation of DIF-SR is based on Recbole. You can run the experiments using following commands:
```bash
python run_recbole.py --model=SASRecD --dataset='yelp' --config_files='configs/yelp.yaml'
```

### Results
Recall@20 = 0.1006

NDCG@20 = 0.0493

### Logs
```bash
05 Apr 03:36    INFO  
General Hyper Parameters:
gpu_id = 3
use_gpu = True
seed = 212
state = INFO
reproducibility = True
data_path = dataset/yelp
show_progress = True
save_dataset = False
save_dataloaders = False
benchmark_filename = None

Training Hyper Parameters:
checkpoint_dir = saved/yelp/SASRecD_4_8_['categories']_[64]_gate_[10]_50_2048_256_0.0001_212
epochs = 200
train_batch_size = 2048
learner = adam
learning_rate = 0.0001
eval_step = 2
stopping_step = 10
clip_grad_norm = None
weight_decay = 0.0
loss_decimal_place = 4

Evaluation Hyper Parameters:
eval_args = {'split': {'LS': 'valid_and_test'}, 'group_by': 'user', 'order': 'TO', 'mode': 'full'}
metrics = ['Recall', 'NDCG']
topk = [5, 10, 20]
valid_metric = Recall@10
valid_metric_bigger = True
eval_batch_size = 256
metric_decimal_place = 4

Dataset Hyper Parameters:
field_separator = 	
seq_separator =  
USER_ID_FIELD = user_id
ITEM_ID_FIELD = business_id
RATING_FIELD = stars
TIME_FIELD = date
seq_len = None
LABEL_FIELD = label
threshold = None
NEG_PREFIX = neg_
load_col = {'inter': ['review_id', 'user_id', 'business_id', 'stars', 'useful', 'funny', 'cool', 'date'], 'item': ['business_id', 'item_name', 'address', 'city', 'state', 'postal_code', 'latitude', 'longitude', 'item_stars', 'item_review_count', 'is_open', 'categories']}
unload_col = None
unused_col = None
additional_feat_suffix = None
rm_dup_inter = None
val_interval = {'date': '[1546264800,1577714400]'}
filter_inter_by_user_or_item = True
user_inter_num_interval = [5,inf)
item_inter_num_interval = [5,inf)
alias_of_user_id = None
alias_of_item_id = None
alias_of_entity_id = None
alias_of_relation_id = None
preload_weight = None
normalize_field = None
normalize_all = None
ITEM_LIST_LENGTH_FIELD = item_length
LIST_SUFFIX = _list
MAX_ITEM_LIST_LENGTH = 50
POSITION_FIELD = position_id
HEAD_ENTITY_ID_FIELD = head_id
TAIL_ENTITY_ID_FIELD = tail_id
RELATION_ID_FIELD = relation_id
ENTITY_ID_FIELD = entity_id

Other Hyper Parameters: 
neg_sampling = None
repeatable = True
n_layers = 4
n_heads = 8
hidden_size = 256
inner_size = 256
hidden_dropout_prob = 0.5
attn_dropout_prob = 0.3
hidden_act = gelu
layer_norm_eps = 1e-12
initializer_range = 0.02
selected_features = ['categories']
pooling_mode = sum
loss_type = CE
MODEL_TYPE = ModelType.SEQUENTIAL
attribute_hidden_size = [64]
weight_sharing = not
fusion_type = gate
lamdas = [10]
attribute_predictor = linear
predictor_source = item
behavior_feature = not
MODEL_INPUT_TYPE = InputType.POINTWISE
eval_type = EvaluatorType.RANKING
device = cuda
train_neg_sample_args = {'strategy': 'none'}
eval_neg_sample_args = {'strategy': 'full', 'distribution': 'uniform'}


05 Apr 03:37    INFO  yelp
The number of users: 30500
Average actions of users: 10.399750811502017
The number of items: 20069
Average actions of items: 15.805361769982062
The number of inters: 317182
The sparsity of the dataset: 99.94818172387231%
Remain Fields: ['review_id', 'user_id', 'business_id', 'stars', 'useful', 'funny', 'cool', 'date', 'item_name', 'address', 'city', 'state', 'postal_code', 'latitude', 'longitude', 'item_stars', 'item_review_count', 'is_open', 'categories']
05 Apr 03:37    INFO  [Training]: train_batch_size = [2048] negative sampling: [None]
05 Apr 03:37    INFO  [Evaluation]: eval_batch_size = [256] eval_args: [{'split': {'LS': 'valid_and_test'}, 'group_by': 'user', 'order': 'TO', 'mode': 'full'}]
05 Apr 03:37    INFO  SASRecD(
  (item_embedding): Embedding(20069, 256, padding_idx=0)
  (position_embedding): Embedding(50, 256)
  (feature_embed_layer_list): ModuleList(
    (0): FeatureSeqEmbLayer()
  )
  (trm_encoder): DeTransformerEncoder(
    (layer): ModuleList(
      (0): DeTransformerLayer(
        (multi_head_attention): DeMultiHeadAttention(
          (query): Linear(in_features=256, out_features=256, bias=True)
          (key): Linear(in_features=256, out_features=256, bias=True)
          (value): Linear(in_features=256, out_features=256, bias=True)
          (query_p): Linear(in_features=256, out_features=256, bias=True)
          (key_p): Linear(in_features=256, out_features=256, bias=True)
          (query_layers): ModuleList(
            (0): Linear(in_features=64, out_features=64, bias=True)
          )
          (key_layers): ModuleList(
            (0): Linear(in_features=64, out_features=64, bias=True)
          )
          (fusion_layer): VanillaAttention(
            (projection): Sequential(
              (0): Linear(in_features=50, out_features=50, bias=True)
              (1): ReLU(inplace=True)
              (2): Linear(in_features=50, out_features=1, bias=True)
            )
          )
          (attn_dropout): Dropout(p=0.3, inplace=False)
          (dense): Linear(in_features=256, out_features=256, bias=True)
          (LayerNorm): LayerNorm((256,), eps=1e-12, elementwise_affine=True)
          (out_dropout): Dropout(p=0.5, inplace=False)
        )
        (feed_forward): FeedForward(
          (dense_1): Linear(in_features=256, out_features=256, bias=True)
          (dense_2): Linear(in_features=256, out_features=256, bias=True)
          (LayerNorm): LayerNorm((256,), eps=1e-12, elementwise_affine=True)
          (dropout): Dropout(p=0.5, inplace=False)
        )
      )
      (1): DeTransformerLayer(
        (multi_head_attention): DeMultiHeadAttention(
          (query): Linear(in_features=256, out_features=256, bias=True)
          (key): Linear(in_features=256, out_features=256, bias=True)
          (value): Linear(in_features=256, out_features=256, bias=True)
          (query_p): Linear(in_features=256, out_features=256, bias=True)
          (key_p): Linear(in_features=256, out_features=256, bias=True)
          (query_layers): ModuleList(
            (0): Linear(in_features=64, out_features=64, bias=True)
          )
          (key_layers): ModuleList(
            (0): Linear(in_features=64, out_features=64, bias=True)
          )
          (fusion_layer): VanillaAttention(
            (projection): Sequential(
              (0): Linear(in_features=50, out_features=50, bias=True)
              (1): ReLU(inplace=True)
              (2): Linear(in_features=50, out_features=1, bias=True)
            )
          )
          (attn_dropout): Dropout(p=0.3, inplace=False)
          (dense): Linear(in_features=256, out_features=256, bias=True)
          (LayerNorm): LayerNorm((256,), eps=1e-12, elementwise_affine=True)
          (out_dropout): Dropout(p=0.5, inplace=False)
        )
        (feed_forward): FeedForward(
          (dense_1): Linear(in_features=256, out_features=256, bias=True)
          (dense_2): Linear(in_features=256, out_features=256, bias=True)
          (LayerNorm): LayerNorm((256,), eps=1e-12, elementwise_affine=True)
          (dropout): Dropout(p=0.5, inplace=False)
        )
      )
      (2): DeTransformerLayer(
        (multi_head_attention): DeMultiHeadAttention(
          (query): Linear(in_features=256, out_features=256, bias=True)
          (key): Linear(in_features=256, out_features=256, bias=True)
          (value): Linear(in_features=256, out_features=256, bias=True)
          (query_p): Linear(in_features=256, out_features=256, bias=True)
          (key_p): Linear(in_features=256, out_features=256, bias=True)
          (query_layers): ModuleList(
            (0): Linear(in_features=64, out_features=64, bias=True)
          )
          (key_layers): ModuleList(
            (0): Linear(in_features=64, out_features=64, bias=True)
          )
          (fusion_layer): VanillaAttention(
            (projection): Sequential(
              (0): Linear(in_features=50, out_features=50, bias=True)
              (1): ReLU(inplace=True)
              (2): Linear(in_features=50, out_features=1, bias=True)
            )
          )
          (attn_dropout): Dropout(p=0.3, inplace=False)
          (dense): Linear(in_features=256, out_features=256, bias=True)
          (LayerNorm): LayerNorm((256,), eps=1e-12, elementwise_affine=True)
          (out_dropout): Dropout(p=0.5, inplace=False)
        )
        (feed_forward): FeedForward(
          (dense_1): Linear(in_features=256, out_features=256, bias=True)
          (dense_2): Linear(in_features=256, out_features=256, bias=True)
          (LayerNorm): LayerNorm((256,), eps=1e-12, elementwise_affine=True)
          (dropout): Dropout(p=0.5, inplace=False)
        )
      )
      (3): DeTransformerLayer(
        (multi_head_attention): DeMultiHeadAttention(
          (query): Linear(in_features=256, out_features=256, bias=True)
          (key): Linear(in_features=256, out_features=256, bias=True)
          (value): Linear(in_features=256, out_features=256, bias=True)
          (query_p): Linear(in_features=256, out_features=256, bias=True)
          (key_p): Linear(in_features=256, out_features=256, bias=True)
          (query_layers): ModuleList(
            (0): Linear(in_features=64, out_features=64, bias=True)
          )
          (key_layers): ModuleList(
            (0): Linear(in_features=64, out_features=64, bias=True)
          )
          (fusion_layer): VanillaAttention(
            (projection): Sequential(
              (0): Linear(in_features=50, out_features=50, bias=True)
              (1): ReLU(inplace=True)
              (2): Linear(in_features=50, out_features=1, bias=True)
            )
          )
          (attn_dropout): Dropout(p=0.3, inplace=False)
          (dense): Linear(in_features=256, out_features=256, bias=True)
          (LayerNorm): LayerNorm((256,), eps=1e-12, elementwise_affine=True)
          (out_dropout): Dropout(p=0.5, inplace=False)
        )
        (feed_forward): FeedForward(
          (dense_1): Linear(in_features=256, out_features=256, bias=True)
          (dense_2): Linear(in_features=256, out_features=256, bias=True)
          (LayerNorm): LayerNorm((256,), eps=1e-12, elementwise_affine=True)
          (dropout): Dropout(p=0.5, inplace=False)
        )
      )
    )
  )
  (ap): ModuleList(
    (0): Linear(in_features=256, out_features=1603, bias=True)
  )
  (LayerNorm): LayerNorm((256,), eps=1e-12, elementwise_affine=True)
  (dropout): Dropout(p=0.5, inplace=False)
  (loss_fct): CrossEntropyLoss()
  (attribute_loss_fct): BCEWithLogitsLoss()
)
Trainable parameters: 7716071
SASRecD
['configs/config_d_yelp.yaml', 'configs/config_training.yaml', 'configs/config_m_SASRecD.yaml']

Train     0:   0%|                                                          | 0/111 [00:00<?, ?it/s]total_loss:16.9830322265625	item_loss:9.925602912902832	attribute_categories_loss:0.7057430148124695

Train     0:   0%|                                | 0/111 [00:00<?, ?it/s, GPU RAM: 14.13 G/23.70 G]
Train     0:   1%|▏                       | 1/111 [00:00<00:57,  1.90it/s, GPU RAM: 14.13 G/23.70 G]total_loss:16.959598541259766	item_loss:9.905494689941406	attribute_categories_loss:0.705410361289978

Train     0:   1%|▏                       | 1/111 [00:00<00:57,  1.90it/s, GPU RAM: 15.20 G/23.70 G]
Train     0:   2%|▍                       | 2/111 [00:00<00:50,  2.18it/s, GPU RAM: 15.20 G/23.70 G]total_loss:16.958240509033203	item_loss:9.910136222839355	attribute_categories_loss:0.7048103213310242

Train     0:   2%|▍                       | 2/111 [00:01<00:50,  2.18it/s, GPU RAM: 15.20 G/23.70 G]
Train     0:   3%|▋                       | 3/111 [00:01<00:46,  2.31it/s, GPU RAM: 15.20 G/23.70 G]total_loss:16.957307815551758	item_loss:9.91263198852539	attribute_categories_loss:0.7044675350189209

Train     0:   3%|▋                       | 3/111 [00:01<00:46,  2.31it/s, GPU RAM: 15.20 G/23.70 G]
Train     0:   4%|▊                       | 4/111 [00:01<00:45,  2.34it/s, GPU RAM: 15.20 G/23.70 G]total_loss:16.960308074951172	item_loss:9.920580863952637	attribute_categories_loss:0.7039727568626404

Train     0:   4%|▊                       | 4/111 [00:02<00:45,  2.34it/s, GPU RAM: 15.20 G/23.70 G]
Train     0:   5%|█                       | 5/111 [00:02<00:44,  2.40it/s, GPU RAM: 15.20 G/23.70 G]total_loss:16.96025276184082	item_loss:9.92430591583252	attribute_categories_loss:0.7035946846008301

Train     0:   5%|█                       | 5/111 [00:02<00:44,  2.40it/s, GPU RAM: 15.20 G/23.70 G]
Train     0:   5%|█▎                      | 6/111 [00:02<00:43,  2.44it/s, GPU RAM: 15.20 G/23.70 G]total_loss:16.94727325439453	item_loss:9.918614387512207	attribute_categories_loss:0.7028657793998718

Train     0:   5%|█▎                      | 6/111 [00:02<00:43,  2.44it/s, GPU RAM: 15.20 G/23.70 G]
Train     0:   6%|█▌                      | 7/111 [00:02<00:42,  2.46it/s, GPU RAM: 15.20 G/23.70 G]total_loss:16.920330047607422	item_loss:9.89969539642334	attribute_categories_loss:0.7020633816719055
...
Evaluate   :  97%|█████████████████████▎| 116/120 [00:02<00:00, 45.95it/s, GPU RAM: 15.21 G/23.70 G]
Evaluate   :  97%|█████████████████████▎| 116/120 [00:02<00:00, 45.95it/s, GPU RAM: 15.21 G/23.70 G]
Evaluate   :  97%|█████████████████████▎| 116/120 [00:02<00:00, 45.95it/s, GPU RAM: 15.21 G/23.70 G]
Evaluate   : 100%|██████████████████████| 120/120 [00:02<00:00, 47.57it/s, GPU RAM: 15.21 G/23.70 G]
05 Apr 04:46    INFO  best valid : {'recall@5': 0.0485, 'recall@10': 0.0729, 'recall@20': 0.1075, 'ndcg@5': 0.0337, 'ndcg@10': 0.0416, 'ndcg@20': 0.0503}
05 Apr 04:46    INFO  test result: {'recall@5': 0.0471, 'recall@10': 0.0671, 'recall@20': 0.1006, 'ndcg@5': 0.0345, 'ndcg@10': 0.0409, 'ndcg@20': 0.0493}

```
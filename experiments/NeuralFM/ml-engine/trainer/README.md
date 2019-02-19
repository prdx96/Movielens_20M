

実行コマンド

```
BUCKET_NAME=dev-michishita-recommend
REGION=us-central1

HPTUNING_CONFIG=trainer/config/hptuning_config.yaml
TRAINDATA_FILEPATH=gs://$BUCKET_NAME/data/

JOB_NAME="recommend_nfm_$(date +%Y%m%d_%H%M%S)"
OUTPUT_PATH=gs://$BUCKET_NAME/$JOB_NAME

gcloud ml-engine jobs submit training $JOB_NAME \
  --job-dir $OUTPUT_PATH \
  --runtime-version 1.8 \
  --module-name trainer.task \
  --package-path trainer/ \
  --region $REGION \
  --python-version 3.5 \
  -- \
  --path $TRAINDATA_FILEPATH \
  --train-steps 128 \
  --batch_size 64000 \
  --epoch 5
```

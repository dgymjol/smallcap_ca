# PYTHONPATH=/data/public/polar/ws/smallcap python src/extract_features_i2t2.py

experiment_name="$1"

python train.py --experiments_dir ${experiment_name} --features_dir i2t_features2
mkdir "${experiment_name}/results"

for var in 8856 17712 26568 35424 44280 53136 61992 70848 79704 88560
do
  CUDA_VISIBLE_DEVICES=1 python infer.py  --features_path "i2t_features2/val.hdf5" --model_path "${experiment_name}/rag_7M_gpt2" --checkpoint_path checkpoint-${var}
  CUDA_VISIBLE_DEVICES=1 python coco-caption/run_eval.py coco-caption/annotations/captions_valKarpathy.json "${experiment_name}/rag_7M_gpt2/checkpoint-${var}/val_preds.json" > "${experiment_name}/results/val_${var}.txt"

#   CUDA_VISIBLE_DEVICES=1 python infer.py --features_path "i2t_features2/test.hdf5" --model_path "${experiment_name}_finetuning_${lr}/rag_7M_gpt2" --checkpoint_path checkpoint-${var} --infer_test
#   CUDA_VISIBLE_DEVICES=1 python coco-caption/run_eval.py coco-caption/annotations/captions_testKarpathy.json "${experiment_name}_finetuning_${lr}/rag_7M_gpt2/checkpoint-${var}/test_preds.json" > "${experiment_name}_finetuning_${lr}/results/test_${var}.txt"
done


# # 12. Inference (val set) (If you specify --infer_test inference uses test data, else val data is used.)
# python infer.py --model_path experiments/rag_7M_gpt2 --checkpoint_path checkpoint-15498
# python infer.py --model_path experiments/rag_7M_gpt2 --checkpoint_path checkpoint-17712
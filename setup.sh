

# # https://github.com/dgymjol/smallcap_mlp.git

# # sudo apt-get install libgl1-mesa-glx libglib2.0-0 libgl1
# # sudo apt-get install default-jre

# # git config --global user.email dgymjol@yonsei.ac.kr
# # git config --global user.name dgymjol


# # 1. 
# conda create -n smallcap
# conda activate smallcap

# conda create -n smallcap2 --clone smallcap
# conda activate smallcap
# pip install transformers==4.30.2

# # 2. 
# # for 1080
# # conda install pytorch==1.12.1 torchvision==0.13.1 torchaudio==0.12.1 cudatoolkit=10.2 -c pytorch
# # for 3090
# conda install pytorch==1.12.1 torchvision==0.13.1 torchaudio==0.12.1 cudatoolkit=11.3 -c pytorch


# # 3.
# pip install -r requirements.txt

# # 5. coco_index (( https://drive.google.com/u/0/uc?id=1ZP5I-xbjaNU7cU48C_ctHd95SaA0jBHe&export=download ))
# mkdir datastore
# cd datastore
# wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id=1ZP5I-xbjaNU7cU48C_ctHd95SaA0jBHe' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=1ZP5I-xbjaNU7cU48C_ctHd95SaA0jBHe" -O coco_index && rm -rf /tmp/cookies.txt
# cd cd../

# 6. coco_index_captions.json (( https://drive.google.com/file/d/1BT0Qc6g40fvtnJ_yY0aipfCuCMgu5qaR/view
# wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id=1BT0Qc6g40fvtnJ_yY0aipfCuCMgu5qaR' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=1BT0Qc6g40fvtnJ_yY0aipfCuCMgu5qaR" -O coco_index_captions.json && rm -rf /tmp/cookies.txt


# 7. file dataset_coco.json from here and place it in data/.
# wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id=11t0ThTr48acmIahcSJ63FX2f0rEORQf8' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=11t0ThTr48acmIahcSJ63FX2f0rEORQf8" -O dataset_coco.json && rm -rf /tmp/cookies.txt
# # 6-1. 
# cd coco-caption
# bash get_stanford_models.sh
# cd ../

# # 7. download coco dataset 

# curl -O http://images.cocodataset.org/zips/val2017.zip
# curl -O http://images.cocodataset.org/zips/train2017.zip 
# curl -O http://images.cocodataset.org/zips/test2017.zip 

# unzip val2017.zip
# unzip train2017.zip
# unzip test2017.zip

# mv val2017/*.jpg train2017
# mv test2017/*.jpg train2017

# rm -rf val2017
# rm -rf test2017

# mv train2017 data/images

# # 8.
# pip install git+https://github.com/openai/CLIP.git

# # 9.
# mkdir features
# conda activate smallcap
# python src/extract_features.py

# # 10. (MUST BE transformers 4.30.2)
# conda activate smallcap2
# python src/retrieve_caps.py


# # 11. Model training (MUST BE transformers 4.21.1)
# conda activate smallcap
# CUDA_VISIBLE_DEVICES=0 python train.py

# # 12. Inference (val set) (If you specify --infer_test inference uses test data, else val data is used.)
# python infer.py --model_path experiments/rag_7M_gpt2 --checkpoint_path checkpoint-15498
# python infer.py --model_path experiments/rag_7M_gpt2 --checkpoint_path checkpoint-17712
# python infer.py --model_path experiments/rag_7M_gpt2 --checkpoint_path checkpoint-19926
# python infer.py --model_path experiments/rag_7M_gpt2 --checkpoint_path checkpoint-22140


# # 14. Evaluate
# python coco-caption/run_eval.py coco-caption/annotations/captions_valKarpathy.json experiments/rag_7M_gpt2/checkpoint-15498/val_preds.json
# python coco-caption/run_eval.py coco-caption/annotations/captions_valKarpathy.json experiments/rag_7M_gpt2/checkpoint-17712/val_preds.json
# python coco-caption/run_eval.py coco-caption/annotations/captions_valKarpathy.json experiments/rag_7M_gpt2/checkpoint-19926/val_preds.json
# python coco-caption/run_eval.py coco-caption/annotations/captions_valKarpathy.json experiments/rag_7M_gpt2/checkpoint-22140/val_preds.json

# # 15. test the best checkpoint in val results

# python infer.py --model_path experiments/rag_7M_gpt2 --checkpoint_path checkpoint-15498 --infer_test
# python coco-caption/run_eval.py coco-caption/annotations/captions_testKarpathy.json experiments/rag_7M_gpt2/checkpoint-15498/test_preds.json

# CUDA_VISIBLE_DEVICES=0 python infer.py --model_path experiments/rag_7M_gpt2 --checkpoint_path checkpoint-8856
# CUDA_VISIBLE_DEVICES=0 python infer.py --model_path experiments/rag_7M_gpt2 --checkpoint_path checkpoint-17712
# CUDA_VISIBLE_DEVICES=0 python infer.py --model_path experiments/rag_7M_gpt2 --checkpoint_path checkpoint-26568
# CUDA_VISIBLE_DEVICES=0 python infer.py --model_path experiments/rag_7M_gpt2 --checkpoint_path checkpoint-35424
# CUDA_VISIBLE_DEVICES=0 python infer.py --model_path experiments/rag_7M_gpt2 --checkpoint_path checkpoint-44280
# CUDA_VISIBLE_DEVICES=0 python infer.py --model_path experiments/rag_7M_gpt2 --checkpoint_path checkpoint-53136
# CUDA_VISIBLE_DEVICES=0 python infer.py --model_path experiments/rag_7M_gpt2 --checkpoint_path checkpoint-61992
# CUDA_VISIBLE_DEVICES=0 python infer.py --model_path experiments/rag_7M_gpt2 --checkpoint_path checkpoint-70848
# CUDA_VISIBLE_DEVICES=0 python infer.py --model_path experiments/rag_7M_gpt2 --checkpoint_path checkpoint-79704
# CUDA_VISIBLE_DEVICES=0 python infer.py --model_path experiments/rag_7M_gpt2 --checkpoint_path checkpoint-88560

# CUDA_VISIBLE_DEVICES=0 python coco-caption/run_eval.py coco-caption/annotations/captions_valKarpathy.json experiments/rag_7M_gpt2/checkpoint-8856/val_preds.json > val_result/result_8856.txt
# CUDA_VISIBLE_DEVICES=0 python coco-caption/run_eval.py coco-caption/annotations/captions_valKarpathy.json experiments/rag_7M_gpt2/checkpoint-17712/val_preds.json > val_result/result_17712.txt
# CUDA_VISIBLE_DEVICES=0 python coco-caption/run_eval.py coco-caption/annotations/captions_valKarpathy.json experiments/rag_7M_gpt2/checkpoint-26568/val_preds.json > val_result/result_26568.txt
# CUDA_VISIBLE_DEVICES=0 python coco-caption/run_eval.py coco-caption/annotations/captions_valKarpathy.json experiments/rag_7M_gpt2/checkpoint-35424/val_preds.json > val_result/result_35424.txt
# CUDA_VISIBLE_DEVICES=0 python coco-caption/run_eval.py coco-caption/annotations/captions_valKarpathy.json experiments/rag_7M_gpt2/checkpoint-44280/val_preds.json > val_result/result_44280.txt
# CUDA_VISIBLE_DEVICES=0 python coco-caption/run_eval.py coco-caption/annotations/captions_valKarpathy.json experiments/rag_7M_gpt2/checkpoint-53136/val_preds.json > val_result/result_53136.txt
# CUDA_VISIBLE_DEVICES=0 python coco-caption/run_eval.py coco-caption/annotations/captions_valKarpathy.json experiments/rag_7M_gpt2/checkpoint-61992/val_preds.json > val_result/result_61992.txt
# CUDA_VISIBLE_DEVICES=0 python coco-caption/run_eval.py coco-caption/annotations/captions_valKarpathy.json experiments/rag_7M_gpt2/checkpoint-70848/val_preds.json > val_result/result_70848.txt
# CUDA_VISIBLE_DEVICES=0 python coco-caption/run_eval.py coco-caption/annotations/captions_valKarpathy.json experiments/rag_7M_gpt2/checkpoint-79704/val_preds.json > val_result/result_79704.txt
# CUDA_VISIBLE_DEVICES=0 python coco-caption/run_eval.py coco-caption/annotations/captions_valKarpathy.json experiments/rag_7M_gpt2/checkpoint-88560/val_preds.json > val_result/result_88560.txt

# best : 53136 (CIDER : 117.14)
python infer.py --model_path experiments/rag_7M_gpt2 --checkpoint_path checkpoint-53136 --infer_test
python coco-caption/run_eval.py coco-caption/annotations/captions_testKarpathy.json experiments/rag_7M_gpt2/checkpoint-53136/test_preds.json > org_text.txt